# Bonus: an example anonymizer
# @see https://docs.smith.langchain.com/observability/how_to_guides/mask_inputs_outputs#rule-based-masking-of-inputs-and-outputs
# TODO: find a way to pass the langsmith client automatically on all calls
# Currently it must be passed to the graph via "langsmith_extra"
from langsmith.anonymizer import create_anonymizer
from langsmith import Client
import re

# create anonymizer from list of regex patterns and replacement values
anonymizer = create_anonymizer([
    { "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}", "replace": "<email-address>" },
    { "pattern": r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "replace": "<UUID>" }
])

# or create anonymizer from a function
email_pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+.[a-zA-Z]{2,}")
uuid_pattern = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

anonymizer = create_anonymizer(
    lambda text: email_pattern.sub("<email-address>", uuid_pattern.sub("<UUID>", text))
)

langsmith_client = Client(anonymizer=anonymizer)