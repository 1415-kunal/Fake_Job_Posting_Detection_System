
import os

# Temporary diagnostic workaround for the OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from gliner import GLiNER

print("Loading GLiNER model...")

model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

text = "Senior AI Engineer at TechNova in Pune, Maharashtra, India."

labels = ["job title", "company name", "location"]

entities = model.predict_entities(
    text,
    labels,
    threshold=0.4,
)

print("\nExtracted entities:")

for entity in entities:
    print(
        f"{entity['text']} -> "
        f"{entity['label']} "
        f"({entity['score']:.2f})"
    )
