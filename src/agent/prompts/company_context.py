"""Static company context injected into every agent prompt.

SAMPLE DATA. Tidewater Labs is a fictional company invented for this public
version of the project. Nothing below describes a real business. Replace the
three constants with your own company before using the system for real advice.

- ``COMPANY_NAME`` and ``COMPANY_DESCRIPTION`` are inserted into the prompt
  templates in this package when they are imported.
- ``COMPANY_CONTEXT`` is the baseline company description that the
  ``context_loader`` node augments with live data (hypotheses, decisions) before
  passing it downstream. All specialist and synthesis prompts receive it.
"""

COMPANY_NAME = "Tidewater Labs"

#: Completes the sentence "<name> is ..." and "advising <name>, ...".
COMPANY_DESCRIPTION = "a startup selling inventory forecasting software to regional grocery chains"

COMPANY_CONTEXT = f"""## About {COMPANY_NAME}

**{COMPANY_NAME}** is a fictional startup used as sample data. It sells inventory \
forecasting software to regional grocery chains. The product reads point-of-sale and \
delivery data and suggests daily order quantities for perishable goods.

**One-liner**: "Order forecasts for regional grocers, so less fresh food is thrown away."

**Team**: Two co-founders, one technical and one commercial.

**Target customer**: Regional grocery chains with 10 to 60 stores. The first contact is \
usually the director of operations or the head of fresh food purchasing.

**Product today**: A web dashboard with daily order suggestions per store and per \
category, a spreadsheet export, and a weekly report on waste and out-of-stock items.

**Current state**:
- Two unpaid pilots running, each in a handful of stores
- No revenue yet, and pricing has not been tested
- Customer interviews are ongoing

Replace this block with your own company description."""


def fill_company(template: str) -> str:
    """Insert the company name and description into a prompt template.

    Templates mark the two values with ``[[COMPANY_NAME]]`` and
    ``[[COMPANY_DESCRIPTION]]``. Braces in the values are escaped because the
    templates are later filled with ``str.format``.

    Args:
        template: Prompt template text containing the markers.

    Returns:
        The template with both markers replaced.
    """

    def _escape(value: str) -> str:
        return value.replace("{", "{{").replace("}", "}}")

    return template.replace("[[COMPANY_NAME]]", _escape(COMPANY_NAME)).replace(
        "[[COMPANY_DESCRIPTION]]", _escape(COMPANY_DESCRIPTION)
    )
