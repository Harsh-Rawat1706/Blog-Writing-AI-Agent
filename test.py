from guardrails import Guard, OnFailAction
from guardrails_ai.nsfw_text import NSFWText

guard = Guard().use(
    NSFWText(
        threshold=0.80,
        validation_method="sentence",
        on_fail=OnFailAction.EXCEPTION,
        use_local=True,
    )
)

try:
    result = guard.validate(
        "Artificial intelligence is transforming education."
    )
    print("Guardrails validation successful!")
    print(result)

except Exception as e:
    print("Guardrails validation failed:", e)