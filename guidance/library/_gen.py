import regex as regex_module
import logging
import guidance
from ._silent import silent
from .._grammar import select
from ._zero_or_more import zero_or_more
from .._grammar import commit_point
from ._any_char import any_char
from .._grammar import capture
from ._regex import regex as regex_grammar
from .._grammar import token_limit
from .._grammar import with_temperature
from .._grammar import model_variable
from ._tool import Tool
from ._block import block

logger = logging.getLogger(__name__)

# TODO: make this stateless!
@guidance(stateless=lambda *args, **kwargs: kwargs.get("tools", None) is None) # TODO: uncomment this once we get temperature stateless
def gen(lm, name=None, *, max_tokens=1000, list_append=False, regex=None,
        tools=None, hide_tool_call=False, stop=None, stop_regex=None, suffix="", n=1, temperature=0.0, top_p=1.0,
        save_stop_text=False):
    """ Generate a set of tokens until a given stop criteria has been met.

    This function is a useful utility that can allow you to specify most grammars used by typical
    LM generation programs. It also has the added ability to interleave generation with tool calls.

    Parameters
        ----------
        name : str or None
            If this is not None then the the results of the generation will be saved as a variable on
            the Model object (so you can access the result as `lm["var_name"]`).

        max_tokens : int
            The maximum number of generation tokens we should use. Note that this limit is not exact when
            regular expression pattern constraints are present, but guidance does attempt to end the generation
            as soon as possible while keeping the regex constraints satisfied.

        list_append : bool
            If this is True then the results saved to `lm[name]` will not be written directly but rather appended
            to a list (if no list with the current name is present one will be created). This is useful for
            building lists inside python loops.
        
        regex : str or None
            This is a regular expression that will be used to constrain the generation. The model is only allowed
            to generate tokens that match this regular expression. Note that for variable length expressions the
            model is free to continue the expression after a complete match, but generation will terminate as soon
            as the model generates anything that does not match the pattern (this ending behavior may change a bit we
            update guidance to maintain the grammar parsing state between calls).

        stop : str or list or None
            The stop string (or list of strings) we should use for terminating this generation segment.

        stop_regex : str or list or None
            The stop regular expression (or list of regular expressions) we should use for terminating this generation segment.

        save_stop_text : bool or str
            If True then this saves the captured stop text or regex into a variable of the name `str(name) + "_stop_text"`. If
            a string is given then the captured stop text is saved under that name.

        temperature : float
            The temperature to use during this generation call. Note that when parsing ambiguous grammars that include
            multiple conflicting temperatures (for example from multiple possible `gen` calls inside a `select`) the highest
            temperature of all options is used by the model (since we only want to run the model once, not once for every
            possible parse path).

        top_p : float
            TODO! Will control the models top_p generation parameter, but has been yet been implemented beyond top_p=1.0.

        n : int
            TODO! Will control the number of parallel generation calls made during gen.

        tools : Tool or list or None
            A list of guidance.Tool or python functions (which will be converted to guidance.Tool)

        hide_tool_call : bool
            Controls if we should hide the text generated by the model to trigger a tool call. You may want to hide the tool
            call from the model's context if you plan to change it's format after the call is made.
    """
    # TODO: expand the tools doc string
    assert n == 1, "We still need to add support for n>1! Consider putting your gen call in a loop for now."
    assert top_p == 1, "We still need to add support for top_p != 1!"
    
    logger.debug(f'start gen(name="{name}")')

    # set stream if we are interactive
    # if stream_tokens is None and not lm.is_silent() and n == 1:
    #     stream_tokens = True

    # use the suffix as the stop string if not otherwise specified
    # TODO: still need to make suffix work with grammars
    # eos_token = lm.eos_token.decode('utf8')
    if stop is None and stop_regex is None and suffix != "":
        stop = suffix
    # if stop is None and stop_regex is None and getattr(lm, "suffix", False):
    #     if lm.suffix.startswith("\n"):
    #         stop = "\n"
    #     elif lm.suffix.startswith('"') and str(lm).endswith('"'):
    #         stop = '"'
    #     elif lm.suffix.startswith("'") and str(lm).endswith("'"):
    #         stop = "'"

    # fall back to stopping at the EOS token
    if stop is not False:
        if stop is None:
            stop = []
        if isinstance(stop, str):
            stop = [stop]
        if regex is None:
            stop.append(model_variable('default_end_patterns'))

        if stop_regex is None:
            stop_regex = []
        if isinstance(stop_regex, str):
            stop_regex = [stop_regex]
        stop_regex = [regex_grammar(x) for x in stop_regex]

    # This needs to be here for streaming
    # if name is not None and not list_append:
    #     lm[name] = ""
    
    # define the generation pattern
    if regex is not None:
        pattern = regex_grammar(regex)
    else:
        pattern = zero_or_more(any_char())

    tagged_name = "__LIST_APPEND:" + name if list_append and name is not None else name

    # define any capture group for non-tool calls
    if name is not None and tools is None:
        pattern = capture(pattern, name=tagged_name)
    
    # limit the number of tokens
    pattern = token_limit(pattern, max_tokens)
    
    # define the stop pattern
    if stop is False or len(stop + stop_regex) == 0:
        stop_pattern = ''
    else:
        stop_pattern = select(stop + stop_regex)
        if save_stop_text is True:
            save_stop_text = str(name) + "_stop_text"
        if isinstance(save_stop_text, str):
            stop_pattern = capture(stop_pattern, name=save_stop_text)
        stop_pattern = commit_point(stop_pattern, hidden=True)

    # single generation
    start_pos = len(str(lm))
    if tools is not None:
        with block(tagged_name):
            tools = [Tool(callable=x) if not isinstance(x, Tool) else x for x in tools]
            init_token_count = lm.token_count
            gen_grammar = pattern + select([stop_pattern] + [capture(commit_point(x.call_grammar, hidden=hide_tool_call), name=f'tool{i}') for i, x in enumerate(tools)])
            while lm.token_count <= max_tokens + init_token_count:
                lm = lm._run_stateless(gen_grammar, temperature=temperature) # TODO: we should not be using this internal method
                tool_called = False
                for i in range(len(tools)):
                    tool_i = f'tool{i}'
                    if tool_i in lm:
                        tool_called = True
                        lm += tools[i].tool_call()
                        lm = lm.remove(tool_i)
                if not tool_called:
                    lm += suffix
                    break
    elif n == 1:
        lm += with_temperature(pattern + stop_pattern + suffix, temperature)

    logger.debug(f'finish gen')
    return lm


def click_loop_start(id, total_count, echo, color):
    click_script = '''
function cycle_IDVAL(button_el) {
var i = 0;
while (i < 50) {
var el = document.getElementById("IDVAL_" + i);
if (el.style.display == "inline") {
    el.style.display = "none";
    var next_el = document.getElementById("IDVAL_" + (i+1));
    if (!next_el) {
        next_el = document.getElementById("IDVAL_0");
    }
    if (next_el) {
        next_el.style.display = "inline";
    }
    break;
}
i += 1;
}
button_el.innerHTML = (((i+1) % TOTALCOUNT) + 1)  + "/" + TOTALCOUNT;
}
cycle_IDVAL(this);'''.replace("IDVAL", id).replace("TOTALCOUNT", str(total_count)).replace("\n", "")
    out = f'''<div style='background: rgba(255, 255, 255, 0.0); border-radius: 4px 0px 0px 4px; border: 1px solid {color}; border-right: 0px; padding-left: 3px; padding-right: 3px; user-select: none; color: {color}; display: inline; font-weight: normal; cursor: pointer' onClick='{click_script}'>1/{total_count}</div>'''
    out += f"<div style='display: inline;' id='{id}_0'>"
    return "<||_html:" + out + "_||>"

def click_loop_mid(id, index, echo):
    alpha = 1.0 if not echo else 0.5
    out = f"</div><div style='display: none; opacity: {alpha}' id='{id}_{index}'>"
    return "<||_html:" + out + "_||>"

@guidance
def gen_line(lm, *args, **kwargs):
    return lm.gen(*args, suffix='\n', **kwargs)

@guidance
def gen_quote(lm, name=None, quote='"', *args, **kwargs):
    return lm(quote).gen(*args,name=name, suffix=quote, **kwargs)

@guidance
def will_gen(lm, stop=None, stop_regex=None, ignore_spaces=False, max_tokens=30):
    # this is obviously not the right implementation, just here so we can explore
    if stop and not isinstance(stop, list):
        stop = [stop]
    if stop_regex and not isinstance(stop_regex, list):
        stop_regex = [stop_regex]
    assert (stop is not None) or (stop_regex is not None)
    if not stop:
        stop = []
    if not stop_regex:
        stop_regex = []
    regexes = [regex_module.escape(x) for x in stop + stop_regex]
    optional_space = '\\s*' if ignore_spaces else ''
    pattern = regex_module.compile(f'{optional_space}({"|".join(regexes)})')
    lm2 = lm
    with silent():
        for _ in range(max_tokens):
            lm2 += gen('temp_variable', list_append=True, max_tokens=1)
            if not lm2['temp_variable'] or not pattern.match(''.join(lm2['temp_variable']), partial=True):
                return False
            if pattern.match(''.join(lm2['temp_variable']), partial=False):
                return True
    return False

@guidance
def call_tool(lm, tool):
    return lm + tool.call_grammar + tool.tool_call()