from intent_engine.intent_classifier import classify_intent
from intent_engine.entity_extractor import extract_entities

raw_query = "Which Agniveers haven't returned their kit?"
entities = extract_entities(raw_query)
intent = classify_intent(raw_query, entities)
print("Resolved Intent:", intent)
