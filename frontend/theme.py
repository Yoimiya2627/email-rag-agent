"""Quiet personal mailbox UI: a light rail, an undivided conversation.

Plan: paper #FFFFFF, rail #F6F7F9, ink #273349, muted #667389,
blue #3568D4, rule #E6E9EF. DengXian headings, Microsoft YaHei body,
Segoe UI utility text; installed fonts with local fallbacks.
Compared [top navigation / conversation] with [rail | conversation].
The rail keeps chat, mail and settings stable. Empty: invitation + composer;
active: messages + bottom composer. The envelope wordmark is the signature.
Critique: remove postage decoration, dark rail, repeated branding and colored
cards. Whitespace groups tasks; popovers/tabs hold details. No animation.
"""

_CSS = """
<style>
:root {
 --mail-paper:#FFFFFF; --mail-canvas:#F6F7F9; --mail-ink:#273349;
 --mail-muted:#667389; --mail-action:#3568D4; --mail-rule:#E6E9EF;
 --mail-action-soft:#EEF3FE;
 --mail-body-font:"Microsoft YaHei","Segoe UI",sans-serif;
 --mail-display-font:"DengXian","Microsoft YaHei",sans-serif;
 --mail-utility-font:"Segoe UI","Microsoft YaHei",sans-serif;
}
[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stHeader"] {background:var(--mail-paper);color:var(--mail-ink)}
[data-testid="stAppViewContainer"] {font:14px/1.7 var(--mail-body-font)}
[data-testid="stMainBlockContainer"] {max-width:1400px;padding:3rem 2.5rem 2rem}
[data-testid="stMarkdownContainer"] p,[data-testid="stText"],[data-testid="stWidgetLabel"] p,[data-testid="stCaptionContainer"] p {font:14px/1.7 var(--mail-body-font);color:var(--mail-ink)}
[data-testid="stMarkdownContainer"] :is(h1,h2,h3) {font-family:var(--mail-display-font);color:var(--mail-ink);font-weight:600;line-height:1.4;letter-spacing:0}
[data-testid="stMarkdownContainer"] h1 {font-size:23px;padding:0 0 .35rem}
[data-testid="stMarkdownContainer"] h2 {font-size:20px}
[data-testid="stMarkdownContainer"] h3 {font-size:17px}
[data-testid="stCaptionContainer"] p {color:var(--mail-muted);font-size:12px}
[data-testid="stMarkdownContainer"] a {color:var(--mail-action)}
[data-testid="stMarkdownContainer"] hr {border:0;border-top:1px solid var(--mail-rule);margin:.8rem 0}

/* Native mobile navigation, error messages and service controls remain usable. */
[data-testid="stSidebar"] {width:216px!important;min-width:216px!important;max-width:216px!important;background:var(--mail-canvas);color:var(--mail-ink);border-right:1px solid var(--mail-rule)}
[data-testid="stSidebarContent"] {padding:0 14px}
[data-testid="stSidebarUserContent"] {padding:.3rem .125rem 1rem;margin:0;width:100%}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {min-height:calc(100dvh - 90px);gap:1rem}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_account) {margin-top:auto}
.mail-brand {display:flex;align-items:center;gap:9px;margin:0 0 .6rem;color:var(--mail-ink)}
.mail-brand svg {width:25px;height:25px;color:var(--mail-action);flex:none}
.mail-brand__name {white-space:nowrap;font:600 18px/1.5 var(--mail-display-font)}
.st-key-mail_nav [role="radiogroup"] {gap:4px}
.st-key-workspace_page {width:100%}
.st-key-mail_nav [role="radiogroup"] > label {width:100%;min-height:42px;padding:8px 12px;margin:0;border-radius:7px;gap:10px;color:var(--mail-muted)}
/* Inputs remain accessible by keyboard; only the circular art is removed. */
.st-key-mail_nav [role="radiogroup"] > label > div:first-child {display:none}
.st-key-mail_nav [role="radiogroup"] > label [data-testid="stMarkdownContainer"] p {color:var(--mail-muted);font-size:14px;line-height:1.5}
.st-key-mail_nav [role="radiogroup"] > label:has(input:checked) {background:var(--mail-action-soft)}
.st-key-mail_nav [role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p {color:var(--mail-action);font-weight:600}
.st-key-mail_nav [role="radiogroup"] > label:hover {background:var(--mail-rule)}
.st-key-mail_nav [role="radiogroup"] > label:focus-within {outline:2px solid var(--mail-action);outline-offset:2px}
.st-key-sidebar_account {margin-top:auto;padding-top:1rem;border-top:1px solid var(--mail-rule)}
.st-key-sidebar_account [data-testid="stVerticalBlock"] {gap:.4rem}
.st-key-sidebar_account [data-testid="stPopover"] button {border:0;padding:0 .1rem;color:var(--mail-muted);background:transparent;min-height:30px}
[data-testid="stSidebar"] [data-testid="stText"] {overflow-wrap:anywhere;font-size:13px}

/* One border style, one action color and a visible keyboard focus. */
:is([data-testid="stTextInput"],[data-testid="stTextArea"],[data-testid="stNumberInput"]) :is(input,textarea) {font:14px/1.6 var(--mail-body-font);color:var(--mail-ink)}
:is([data-testid="stTextInput"],[data-testid="stTextArea"],[data-testid="stNumberInput"]) > div,[data-baseweb="select"] > div {border-color:var(--mail-rule);border-radius:7px;background:var(--mail-canvas);color:var(--mail-ink)}
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"] {background:var(--mail-action);border:1px solid var(--mail-action);color:var(--mail-paper);border-radius:7px;min-height:38px}
[data-testid="stBaseButton-primary"]:hover,[data-testid="stBaseButton-primaryFormSubmit"]:hover {filter:brightness(.94);color:var(--mail-paper)}
[data-testid="stBaseButton-primary"] p,[data-testid="stBaseButton-primaryFormSubmit"] p {color:inherit}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-secondaryFormSubmit"] {background:var(--mail-paper);border:1px solid var(--mail-rule);color:var(--mail-ink);border-radius:7px;min-height:36px}
[data-testid="stBaseButton-secondary"]:hover,[data-testid="stBaseButton-secondaryFormSubmit"]:hover {background:var(--mail-action-soft);border-color:var(--mail-action);color:var(--mail-action)}
[data-testid="stMain"] button:disabled {opacity:.5}
[data-testid="stExpander"] > details {border-color:var(--mail-rule);border-radius:7px;background:var(--mail-paper)}
[data-testid="stForm"] {border-color:var(--mail-rule)}
:is(button,a,input,textarea,select,[role="radio"],[role="tab"]):focus-visible {outline:2px solid var(--mail-action);outline-offset:3px}
[data-baseweb="tab-highlight"] {background-color:var(--mail-action)}
[data-baseweb="tab"][aria-selected="true"] {color:var(--mail-action)}
[data-testid="stMetricValue"] {font:600 24px/1.4 var(--mail-display-font)}

/* Empty conversations group the invitation and composer. */
[data-testid="stMainBlockContainer"]:has(.st-key-assistant_header) {max-width:940px;padding-top:2.5rem}
.st-key-assistant_header {padding-bottom:.75rem}
.st-key-assistant_header [data-testid="stVerticalBlock"] {gap:0}
.st-key-assistant_header [data-testid="stPopover"] button {border:0;color:var(--mail-muted)}
.st-key-assistant_header [data-testid="stHorizontalBlock"] {flex-wrap:nowrap;gap:.75rem}
.st-key-assistant_header [data-testid="stColumn"]:first-child {min-width:0}
.st-key-assistant_header [data-testid="stColumn"]:last-child {min-width:78px}
.st-key-assistant_welcome {padding:clamp(2rem,14vh,9rem) 0 1.25rem;text-align:center}
.st-key-assistant_welcome [data-testid="stVerticalBlock"] {gap:.3rem}
.st-key-assistant_welcome [data-testid="stMarkdownContainer"] h3 {font-size:30px;font-weight:500;text-align:center;padding:0 0 .6rem}
.st-key-assistant_welcome [data-testid="stMarkdownContainer"] p {font-size:14px;color:var(--mail-muted);text-align:center}
.st-key-assistant_start {max-width:680px;margin:0 auto}
.st-key-assistant_start [data-testid="stElementContainer"]:has([data-testid="stButton"]) {align-self:center}
.st-key-assistant_start [data-testid="stButton"] {text-align:center}
.st-key-assistant_start [data-testid="stButton"] button {border:0;color:var(--mail-muted);background:transparent;font-size:13px}
.st-key-assistant_history [data-testid="stChatMessage"] {background:transparent;padding:1.15rem 0;border-radius:0;gap:12px}
[data-testid^="stChatMessageAvatar"] {background:var(--mail-action-soft);color:var(--mail-action)}
[data-testid="stBottomBlockContainer"] {max-width:940px;padding:.75rem 2.5rem 1.2rem}
[data-testid="stBottom"],[data-testid="stBottom"] > div {background:var(--mail-paper)}
[data-testid="stChatInput"] {background:var(--mail-canvas);border:1px solid var(--mail-rule);border-radius:16px;padding:.55rem .7rem}
[data-testid="stChatInput"]:focus-within {border-color:var(--mail-action);box-shadow:0 0 0 2px var(--mail-action-soft)}
[data-testid="stChatInput"] textarea {font:15px/1.7 var(--mail-body-font)}
[data-testid="stChatInput"] [data-baseweb="textarea"],[data-testid="stChatInput"] textarea {background:transparent}
.st-key-assistant_start [data-testid="stChatInput"] textarea {min-height:64px}
[data-testid="stChatInputSubmitButton"] {color:var(--mail-action)}

/* Mail reading uses one list/reader boundary, not a stack of cards. */
.st-key-mail_header {padding-bottom:.6rem}
.st-key-mail_list,.st-key-mail_reader {min-width:0;background:var(--mail-paper)}
.st-key-mail_list {padding:.75rem 1.25rem .75rem 0;border-right:1px solid var(--mail-rule)}
.st-key-mail_reader {padding:.75rem 0 .75rem 1rem}
.st-key-mail_list [data-testid="stVerticalBlock"],.st-key-mail_reader [data-testid="stVerticalBlock"] {gap:.7rem}
.mail-subject {overflow-wrap:anywhere}
.mail-body {white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.9 var(--mail-body-font);color:var(--mail-ink)}
.mail-address {overflow-wrap:anywhere;color:var(--mail-muted);font:12px/1.7 var(--mail-body-font)}
.mail-address b {font-weight:500;display:inline-block;min-width:54px}
.mail-table-row {display:flex;flex-wrap:wrap;gap:1rem;margin:.4rem 0;padding:.4rem 0;border-bottom:1px solid var(--mail-rule);white-space:pre-wrap}
.mail-table-cell {flex:1 1 120px;min-width:0;overflow-wrap:anywhere}
.mail-table-cell:empty {display:none}
.mail-table-cell small {display:block;color:var(--mail-muted);font-size:12px}
.st-key-mail_rows [class*="st-key-mail_row_"] {padding:.15rem 0 .65rem;border-bottom:1px solid var(--mail-rule)}
.st-key-mail_rows [data-testid="stButton"] button {justify-content:flex-start;text-align:left;min-height:2.6rem}
.st-key-mail_rows [data-testid="stBaseButton-primary"] {background:var(--mail-action-soft);color:var(--mail-action);border-color:transparent}
.st-key-mail_rows [data-testid="stBaseButton-secondary"] {border:0;padding-left:.2rem}
.st-key-mail_rows [data-testid="stButton"] p {display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;line-height:1.5;text-align:left;width:100%}
.st-key-mail_rows [data-testid="stButton"] [data-testid="stMarkdownContainer"] {width:100%}
.st-key-mail_rows [data-testid="stCaptionContainer"] p {margin:0;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.st-key-mail_reader textarea:disabled {color:var(--mail-ink);-webkit-text-fill-color:var(--mail-ink);opacity:1;cursor:text}
.st-key-mail_settings,.st-key-settings_tools {max-width:860px;margin:0 auto}
.st-key-mail_settings [data-baseweb="tab-panel"] {padding-top:1.25rem}
.st-key-mail_inbox [data-testid="stForm"] {padding:0 0 1rem}

@media (max-width:1000px) {
 [data-testid="stMainBlockContainer"] {padding:3rem 1.25rem 2rem}
}
@media (max-width:720px) {
 [data-testid="stMainBlockContainer"],[data-testid="stMainBlockContainer"]:has(.st-key-assistant_header) {padding:3rem 1rem 2rem}
 [data-testid="stBottomBlockContainer"] {padding:.5rem 1rem 1rem}
 .st-key-assistant_welcome {padding-top:10vh}
 .st-key-assistant_welcome [data-testid="stMarkdownContainer"] h3 {font-size:26px}
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_reader) {flex-wrap:wrap;gap:1rem}
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_reader) > [data-testid="stColumn"] {width:100%;flex:1 1 100%;min-width:0}
 .st-key-mail_list {padding:.75rem 0;border-right:0;border-bottom:1px solid var(--mail-rule)}
 .st-key-mail_reader {padding:.75rem 0}
}
@media (prefers-reduced-motion:reduce) {
 [data-testid="stAppViewContainer"] *,[data-testid="stAppViewContainer"] *::before,[data-testid="stAppViewContainer"] *::after {animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}
}
</style>
"""


def apply_theme(st):
    """Static CSS only; no user-controlled text is interpolated."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_brand(st):
    """Static envelope wordmark with no external assets."""
    st.markdown(
        '<div class="mail-brand">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">'
        '<rect x="2.5" y="4.5" width="19" height="15" rx="3"/>'
        '<path d="m3.5 6 8.5 6.5L20.5 6"/></svg>'
        '<span class="mail-brand__name">邮件助手</span></div>',
        unsafe_allow_html=True,
    )
