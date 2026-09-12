"""Visual identity for 邮序, a local mailbox client.

Plan: 240px deep-sea navigation | list + reader | always-open Agent conversation.
Tokens: sea #173238, canvas #F1F5F6, paper #FFFFFF, action #0F766E,
ink #20343B, muted #657780; rules #DCE5E8. Bahnschrift is reserved for
headings, Segoe UI for reading, and Consolas for small utility labels.
The single signature is a compact 邮 postage mark next to the Chinese name;
provider labels belong to individual accounts, so future QQ accounts fit here.

Plan critique: discard decorative dashboard cards and oversized metrics; this
screen's purpose is choosing and reading mail. Use a real list/reader boundary
instead of unrelated floating cards. Final review: no external assets, no mail
HTML, no hidden controls, visible focus, flexible small-screen panels, and no
animation required. Actual app rendering is checked by the integrating caller.
"""


_CSS = """
<style>
:root {
  --mail-sea: #173238;
  --mail-canvas: #F1F5F6;
  --mail-paper: #FFFFFF;
  --mail-action: #0F766E;
  --mail-ink: #20343B;
  --mail-muted: #657780;
  --mail-rule: #DCE5E8;
  --mail-sidebar-muted: #B7CBCF;
  --mail-action-soft: #E7F3F0;
  --mail-focus: #0F766E;
  --mail-body-font: "Segoe UI", "Microsoft YaHei", sans-serif;
  --mail-display-font: "Bahnschrift", "Microsoft YaHei UI", sans-serif;
  --mail-utility-font: "Consolas", "Microsoft YaHei", monospace;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
  background: var(--mail-canvas);
  color: var(--mail-ink);
}
[data-testid="stAppViewContainer"] {
  font-family: var(--mail-body-font);
  font-size: 14px;
  line-height: 1.6;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1920px;
  padding: 4.5rem 2rem 3rem;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stText"],
[data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p {
  font-family: var(--mail-body-font);
  font-size: 14px;
  line-height: 1.6;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
  font-family: var(--mail-display-font);
  color: var(--mail-ink);
  font-weight: 600;
  line-height: 1.3;
  letter-spacing: -.025em;
}
[data-testid="stMarkdownContainer"] h1 {
  font-size: 28px;
  padding: 0 0 .45rem;
}
[data-testid="stMarkdownContainer"] h2 { font-size: 21px; }
[data-testid="stMarkdownContainer"] h3 { font-size: 17px; }
[data-testid="stCaptionContainer"] { color: var(--mail-muted); }
[data-testid="stWidgetLabel"] { color: var(--mail-muted); }
[data-testid="stMarkdownContainer"] a { color: var(--mail-action); }
[data-testid="stMarkdownContainer"] hr {
  border: 0;
  border-top: 1px solid var(--mail-rule);
  margin: 1rem 0;
}

/* Keep the native sidebar toggle, toolbar, status and error surfaces intact. */
[data-testid="stSidebar"] {
  width: 240px !important;
  min-width: 240px !important;
  max-width: 240px !important;
  background: var(--mail-sea);
  color: var(--mail-paper);
  border-right: 1px solid var(--mail-sea);
}
[data-testid="stSidebarUserContent"] { padding: 1.1rem 1.1rem 1.75rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
  color: var(--mail-paper);
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
  color: var(--mail-sidebar-muted);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
  color: #A9E1D9;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {
  border-top-color: #385057;
}
[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
  color: inherit;
}

.mail-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  margin: 0 0 1.9rem;
  padding-top: .15rem;
  color: var(--mail-paper);
}
.mail-brand__stamp {
  display: grid;
  place-items: center;
  flex: 0 0 42px;
  height: 44px;
  border: 1px dashed var(--mail-sidebar-muted);
  outline: 1px solid #385057;
  outline-offset: 3px;
  color: var(--mail-paper);
  font: 600 19px/1 var(--mail-display-font);
  letter-spacing: -.03em;
}
.mail-brand__name {
  font: 600 23px/1.2 var(--mail-display-font);
  letter-spacing: .12em;
}
.mail-brand__caption {
  margin-top: 5px;
  color: var(--mail-sidebar-muted);
  font: 11px/1.2 var(--mail-utility-font);
  letter-spacing: .16em;
}
.st-key-mail_nav [role="radiogroup"] { gap: 5px; }
.st-key-mail_nav [role="radiogroup"] > label {
  width: 100%;
  min-height: 42px;
  margin: 0;
  padding: 9px 10px;
  border: 1px solid transparent;
  border-radius: 6px;
  color: var(--mail-sidebar-muted);
}
.st-key-mail_nav [role="radiogroup"] > label [data-testid="stMarkdownContainer"] p { color: var(--mail-sidebar-muted); font-size: 14px; }
.st-key-mail_nav [role="radiogroup"] > label:has(input:checked) [data-testid="stMarkdownContainer"] p { color: var(--mail-paper); font-weight: 600; }
.st-key-mail_nav [role="radiogroup"] > label:has(input:checked) > div:first-child { background-color: var(--mail-action); }
.st-key-mail_nav [role="radiogroup"] > label:hover {
  background: #25434A;
  color: var(--mail-paper);
}
.st-key-mail_nav [role="radiogroup"] > label:has(input:checked) {
  border-color: #45636B;
  background: #2B4B52;
  color: var(--mail-paper);
}
.st-key-mail_nav [role="radiogroup"] > label:focus-within {
  outline: 2px solid #A9E1D9;
  outline-offset: 2px;
}

/* These keys are integration hooks, not selectors for private widget IDs. */
.st-key-mail_header {
  padding-bottom: .85rem;
  margin-bottom: .25rem;
  border-bottom: 1px solid var(--mail-rule);
}
.st-key-mail_list,
.st-key-mail_reader,
.st-key-mail_agent {
  min-width: 0;
  padding: 1.1rem 1.15rem;
  border: 1px solid var(--mail-rule);
  border-radius: 8px;
  background: var(--mail-paper);
  color: var(--mail-ink);
}
.st-key-mail_list { border-top: 3px solid var(--mail-action); }
.st-key-mail_reader { padding: 1.25rem 1.5rem; }
.st-key-mail_agent { border-top: 3px solid var(--mail-action); padding: 1.1rem; }
.st-key-mail_agent [data-testid="stChatMessage"] { padding: .7rem; min-width: 0; }
.st-key-mail_agent [data-testid="stChatMessageContent"] { min-width: 0; overflow-wrap: anywhere; }
.st-key-mail_agent [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { font-size: 14px; }
.st-key-mail_agent [data-testid="stChatInput"] { border: 1px solid var(--mail-action); border-radius: 8px; }
.st-key-mail_agent [data-testid="stChatInput"] textarea { font-family: var(--mail-body-font); font-size: 14px; }
.st-key-mail_agent [data-testid="stAlert"] { padding: .65rem .8rem; }
.st-key-mail_agent [data-testid="stCaptionContainer"] p { font-size: 12px; }
.st-key-mail_agent_history { border-top: 1px solid var(--mail-rule); padding-top: .8rem; }
.st-key-mail_workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_agent) > [data-testid="stColumn"] { min-width: 0; }
.mail-body { white-space: pre-wrap; overflow-wrap: anywhere; font: 15px/1.9 var(--mail-body-font); color: var(--mail-ink); }
.mail-address { overflow-wrap: anywhere; color: var(--mail-muted); font: 13px/1.6 var(--mail-body-font); }
.mail-address b { font-weight: 500; display: inline-block; min-width: 54px; }
.mail-subject { overflow-wrap: anywhere; }
.mail-table-row { display:flex; flex-wrap:wrap; gap:1rem; margin:.4rem 0; padding:.4rem 0; border-bottom:1px solid var(--mail-rule); white-space:pre-wrap; }
.mail-table-cell { flex:1 1 120px; min-width:0; overflow-wrap:anywhere; }
.mail-table-cell:empty { display:none; }
.mail-table-cell small { display:block; color:var(--mail-muted); font-size:12px; }
.st-key-mail_rows [class*="st-key-mail_row_"] { padding: .2rem 0 .65rem; border-bottom: 1px solid var(--mail-rule); }
.st-key-mail_rows [data-testid="stButton"] button { justify-content: flex-start; text-align: left; min-height: 2.6rem; }
.st-key-mail_rows [data-testid="stBaseButton-primary"] { background: var(--mail-action-soft); color: var(--mail-action); border-color: var(--mail-action-soft); }
.st-key-mail_rows [data-testid="stBaseButton-secondary"] { border-color: transparent; padding-left: .2rem; }
.st-key-mail_rows [data-testid="stCaptionContainer"] p { font-size: 12px; margin: 0; }
.st-key-mail_list [data-testid="stVerticalBlock"],
.st-key-mail_reader [data-testid="stVerticalBlock"] { gap: .75rem; }
.st-key-mail_list [data-testid="stBaseButton-secondary"] {
  justify-content: flex-start;
  text-align: left;
}
.st-key-mail_reader [data-testid="stText"] {
  overflow-wrap: anywhere;
  line-height: 1.75;
}
.st-key-mail_reader [data-testid="stTextArea"] textarea {
  background: var(--mail-paper);
  font-size: 15px;
  line-height: 1.85;
}
.st-key-mail_reader [data-testid="stTextArea"] textarea:disabled {
  color: var(--mail-ink);
  -webkit-text-fill-color: var(--mail-ink);
  opacity: 1;
  cursor: text;
}
.st-key-mail_reader [data-testid="stJson"] {
  font-family: var(--mail-utility-font);
  font-size: 12px;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
  font-family: var(--mail-body-font);
  font-size: 14px;
  color: var(--mail-ink);
}
[data-testid="stTextInput"] > div,
[data-testid="stTextArea"] > div,
[data-testid="stNumberInput"] > div,
[data-baseweb="select"] > div {
  border-color: var(--mail-rule);
  border-radius: 6px;
  background: var(--mail-paper);
  color: var(--mail-ink);
}
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--mail-action);
  color: var(--mail-paper);
  border: 1px solid var(--mail-action);
  border-radius: 6px;
  font-weight: 600;
  min-height: 38px;
}
[data-testid="stBaseButton-primary"]:hover,
[data-testid="stBaseButton-primaryFormSubmit"]:hover {
  background: #0B605A;
  border-color: #0B605A;
  color: var(--mail-paper);
}
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
  background: var(--mail-paper);
  color: var(--mail-ink);
  border: 1px solid var(--mail-rule);
  border-radius: 6px;
  min-height: 36px;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
  background: var(--mail-action-soft);
  border-color: var(--mail-action);
  color: var(--mail-action);
}
[data-testid="stMain"] button:disabled { opacity: .52; }
[data-testid="stExpander"] > details {
  border-color: var(--mail-rule);
  border-radius: 6px;
  background: var(--mail-paper);
  color: var(--mail-ink);
}
[data-testid="stSidebar"] [data-testid="stExpander"] > details {
  background: #203E45;
  border-color: #385057;
  color: var(--mail-paper);
}
[data-testid="stForm"] { border-color: var(--mail-rule); }
[data-testid="stMetricValue"] {
  font-family: var(--mail-display-font);
  font-size: 23px;
  color: var(--mail-ink);
}

:is(button, a, input, textarea, select, [role="radio"], [role="tab"]):focus-visible {
  outline: 2px solid var(--mail-focus);
  outline-offset: 3px;
}
[data-testid="stSidebar"] :is(button, a, [role="radio"]):focus-visible {
  outline-color: #A9E1D9;
}

/* On narrow screens the assistant comes first, keeping the composer discoverable. */
@media (max-width: 1150px) {
  .st-key-mail_workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_agent) { flex-wrap: wrap; }
  .st-key-mail_workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_agent) > [data-testid="stColumn"] {
    width: 100%; flex: 1 1 100%; min-width: 0;
  }
  .st-key-mail_workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_agent) > [data-testid="stColumn"]:first-child { order: 2; }
  .st-key-mail_workspace [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_agent) > [data-testid="stColumn"]:last-child { order: 1; }
  .st-key-mail_agent_history,
  .st-key-mail_agent [data-testid="stLayoutWrapper"]:has(> .st-key-mail_agent_history) {
    height: 190px !important; flex: 0 0 190px !important;
  }
}
@media (max-width: 1000px) {
  [data-testid="stMainBlockContainer"] { padding: 4.5rem 1rem 2rem; }
  .st-key-mail_list { padding: .9rem; }
  .st-key-mail_reader { padding: 1rem; }
}
@media (max-width: 720px) {
  [data-testid="stMainBlockContainer"] { padding: 4rem .75rem 2rem; }
  [data-testid="stMarkdownContainer"] h1 { font-size: 25px; }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_reader) {
    flex-wrap: wrap;
    gap: 1rem;
  }
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-mail_reader)
    > [data-testid="stColumn"] {
    width: 100%;
    flex: 1 1 100%;
    min-width: 0;
  }
  .st-key-mail_reader { padding: 1rem; }
}
@media (prefers-reduced-motion: reduce) {
  [data-testid="stAppViewContainer"] *,
  [data-testid="stAppViewContainer"] *::before,
  [data-testid="stAppViewContainer"] *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
    scroll-behavior: auto !important;
  }
}
</style>
"""


def apply_theme(st):
    """Apply static, local CSS. No user-controlled text is interpolated."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_brand(st):
    """Render the static wordmark inside the caller's sidebar context."""
    st.markdown(
        '<div class="mail-brand">'
        '<span class="mail-brand__stamp" aria-hidden="true">邮</span>'
        '<div><div class="mail-brand__name">邮序</div>'
        '<div class="mail-brand__caption">MAIL DESK</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
