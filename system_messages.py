"""
system_messages.py
==================
Central catalog of user-facing conversational, warning, error, and entity-not-found
messages for AgniAI.

All messages are designed with:
  - Simple everyday words (no technical jargon)
  - Warm, friendly, and respectful tone
  - Helpful suggestions for guidance
  - Empathetic and clear language
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional

# =============================================================================
# SYSTEM MESSAGE CATALOG
# =============================================================================

DATABASE_CONNECTION_ISSUES: Dict[str, Any] = {
    "main": "I'm having trouble reaching the database right now.",
    "alternatives": [
        "It seems I can't connect to the system at the moment.",
        "The system isn't responding - please give it a moment and try again.",
        "I'm unable to access the information you're looking for right now.",
        "There might be a network issue - could you please check your connection?",
        "I can't seem to get through to the database at this time.",
        "The system appears to be offline or busy right now.",
        "I'm sorry, but I can't retrieve the data at the moment.",
        "It looks like the database is taking a nap - please try again in a few minutes.",
        "I'm having difficulty connecting - maybe the server needs a restart?",
    ],
}

QUERY_PROCESSING_ISSUES: Dict[str, Any] = {
    "main": "I didn't quite understand that - could you please rephrase?",
    "alternatives": [
        "I'm having trouble making sense of your request.",
        "Something about your question is a bit confusing - could you clarify?",
        "I'm not sure I followed that correctly - would you mind explaining again?",
        "That doesn't seem to make complete sense to me - could you help me understand?",
        "I think I missed something - could you please repeat that differently?",
        "Your question seems a bit jumbled - can we try a different approach?",
        "I'm a bit lost - could you point me in the right direction?",
        "That's not quite clear to me - could you provide more detail?",
        "I want to help, but I need a little more guidance on what you need.",
    ],
}

WHEN_CANNOT_FIND_WHAT_LOOKING_FOR: Dict[str, Any] = {
    "main": "I searched everywhere, but I couldn't find what you're looking for.",
    "alternatives": [
        "It seems the information you want isn't in our system yet.",
        "I'm sorry, but I came up empty-handed on that search.",
        "That particular piece of information doesn't seem to exist right now.",
        "I looked high and low, but couldn't locate that data.",
        "Unfortunately, I don't have that information available.",
        "It looks like that record hasn't been created in the system yet.",
        "I'm coming up blank on that search - would you like me to check something else?",
        "I couldn't dig up anything matching your request.",
        "That information seems to be missing from our records.",
    ],
}

WHEN_AGNIVEERS_NOT_FOUND: Dict[str, Any] = {
    "main": "I searched for Agniveers, but couldn't find any in our system.",
    "alternatives": [
        "It looks like no Agniveer records have been entered yet.",
        "I don't see any Agniveers listed in the database.",
        "It appears the Agniveer information hasn't been added to the system yet.",
        "I can't seem to find any Agniveer records - maybe they haven't been entered?",
        "The Agniveer data seems to be empty at the moment.",
        "I looked through the Agniveer records but couldn't find any.",
        "There don't seem to be any Agniveers registered in the system yet.",
        "I'm not seeing any Agniveer profiles in our database.",
        "It looks like we haven't added any Agniveers to the system yet.",
    ],
}

WHEN_COMMANDERS_NOT_FOUND: Dict[str, Any] = {
    "main": "I looked for commanders but couldn't find any assigned yet.",
    "alternatives": [
        "It seems no one has been appointed as a commander so far.",
        "I don't see any commanders assigned to companies or platoons.",
        "The commander positions appear to be empty right now.",
        "I couldn't find anyone holding a commander position in the system.",
        "It looks like commanders haven't been assigned yet.",
        "There are no commanders listed in our records.",
        "I searched for commanders but found none.",
        "It appears the commander roles haven't been filled yet.",
        "No one seems to be assigned as a commander at the moment.",
    ],
}

WHEN_COMPANIES_NOT_FOUND: Dict[str, Any] = {
    "main": "I couldn't find any companies in the system.",
    "alternatives": [
        "It looks like no companies have been created yet.",
        "I don't see any company records available.",
        "The company information seems to be missing.",
        "There are no companies listed in our database.",
        "I searched but couldn't find any companies.",
        "It seems companies haven't been added to the system yet.",
        "I'm not seeing any company records right now.",
        "The company data appears to be empty.",
        "No companies exist in the system at this time.",
    ],
}

WHEN_PLATOONS_NOT_FOUND: Dict[str, Any] = {
    "main": "I couldn't find any platoons in the system.",
    "alternatives": [
        "It looks like no platoons have been created yet.",
        "I don't see any platoon records available.",
        "The platoon information seems to be missing.",
        "There are no platoons listed in our database.",
        "I searched but couldn't find any platoons.",
        "It seems platoons haven't been added to the system yet.",
        "I'm not seeing any platoon records right now.",
        "The platoon data appears to be empty.",
        "No platoons exist in the system at this time.",
    ],
}

WHEN_USERS_NOT_FOUND: Dict[str, Any] = {
    "main": "I couldn't find any users in the system.",
    "alternatives": [
        "It looks like no user accounts have been created yet.",
        "I don't see any users registered in the system.",
        "There are no users listed in our records.",
        "The user information seems to be missing.",
    ],
}

WHEN_BATCHES_NOT_FOUND: Dict[str, Any] = {
    "main": "I couldn't find any batches in the system.",
    "alternatives": [
        "It looks like no batches have been created yet.",
        "I don't see any batch records available.",
        "The batch information seems to be missing.",
        "There are no batches listed in our database.",
    ],
}

WHEN_SPECIFIC_RECORD_NOT_FOUND: Dict[str, Any] = {
    "main": "I searched for that specific record but couldn't find it.",
    "alternatives": [
        "The particular item you're looking for doesn't seem to exist.",
        "I can't locate that specific record you mentioned.",
        "That exact record doesn't appear to be in our system.",
        "I couldn't find the specific one you're referring to.",
    ],
}

DUPLICATE_INFORMATION: Dict[str, Any] = {
    "main": "It looks like this information already exists in the system.",
    "alternatives": [
        "I found a duplicate - this record seems to have been added twice.",
        "That entry appears to already be in our database.",
        "I noticed this is a duplicate of something we already have.",
        "It seems you're trying to add something that's already there.",
    ],
}

MISSING_INFORMATION: Dict[str, Any] = {
    "main": "I'm missing some information to complete your request.",
    "alternatives": [
        "Could you please fill in all the required details?",
        "I need a bit more information to finish this.",
        "Some important details seem to be missing.",
        "Please provide all the necessary information so I can help you.",
    ],
}

INCORRECT_FORMAT: Dict[str, Any] = {
    "main": "I think the format of your input might be a bit off.",
    "alternatives": [
        "The way you entered that doesn't quite match what I'm expecting.",
        "Could you please check the format and try again?",
        "The dates or numbers seem to be in a different format than I need.",
        "I'm having trouble with the format of the information you provided.",
    ],
}

PERMISSION_ISSUES: Dict[str, Any] = {
    "main": "I'm sorry, but you don't seem to have permission to do that.",
    "alternatives": [
        "You might need to ask your administrator for access to this.",
        "I can't complete this - it looks like your access is restricted.",
        "This feature is only available to certain users.",
        "Please check with your supervisor if you need access to this.",
    ],
}

FILE_ATTACHMENT_ISSUES: Dict[str, Any] = {
    "main": "I'm having trouble opening that file.",
    "alternatives": [
        "The file you mentioned doesn't seem to exist.",
        "I can't access that file - maybe it's in a different location?",
        "I couldn't find the file you're referring to.",
        "There seems to be a problem with the file you mentioned.",
    ],
}

GENERAL_HELPFUL_SUGGESTIONS: Dict[str, Any] = {
    "main": "Would you like me to show you what IS available instead?",
    "alternatives": [
        "I can help you add this information if you'd like.",
        "Would you like to try a different search instead?",
        "I can show you all the records that do exist if that helps.",
        "Maybe we can look at this from a different angle?",
        "Would you like me to suggest some alternative options?",
        "I'm here to help - just let me know what you'd like to try next.",
        "Perhaps we can find what you need in a different section?",
        "Would it help if I showed you some examples?",
        "Let me know if you'd like to explore other options.",
    ],
}

ENCOURAGING_AND_KIND_WORDS: Dict[str, Any] = {
    "main": "Please don't worry - I'm here to help you through this.",
    "alternatives": [
        "Your patience is truly appreciated!",
        "Thank you for bearing with me on this.",
        "I hope you're having a good day - let me know if I can do anything else.",
        "Please feel free to ask again if you need anything else.",
        "I'm always happy to help whenever you need me.",
        "You're doing great - let's figure this out together.",
        "I appreciate you taking the time to explain that.",
        "Thank you for your understanding and cooperation.",
        "I'm here whenever you need assistance - just ask!",
    ],
}

FINAL_KIND_CLOSING_MESSAGES: Dict[str, Any] = {
    "main": "Is there anything else I can help you with today?",
    "alternatives": [
        "I hope that information was helpful to you.",
        "Please let me know if you need anything more from me.",
        "Thank you for your time - have a wonderful day!",
        "I'm glad I could help - feel free to come back anytime.",
        "If you need anything else, I'll be right here.",
        "Take care and don't hesitate to ask if you have more questions.",
        "I'm always available when you need assistance.",
        "Thank you for trusting me to help you with this.",
        "Wishing you a great day ahead!",
    ],
}

SUGGESTING_NEXT_STEPS: Dict[str, Any] = {
    "main": "Perhaps we should start by adding some basic information first.",
    "alternatives": [
        "It might help to create the main records before looking for assignments.",
        "Would you like to begin by setting up the companies and platoons?",
        "Maybe we can work together to add the missing information.",
        "Let's start with the basics and build from there.",
        "I can guide you through adding the necessary information if you'd like.",
        "It might be easier if we create the foundation first.",
        "Would you like help setting everything up from scratch?",
        "We can build this step by step if that works for you.",
        "I'd be happy to help you get everything set up properly.",
    ],
}

# Mapping of entity scope to message dict
_ENTITY_NOT_FOUND_MAP: Dict[str, Dict[str, Any]] = {
    "agniveer": WHEN_AGNIVEERS_NOT_FOUND,
    "agniveers": WHEN_AGNIVEERS_NOT_FOUND,
    "commander": WHEN_COMMANDERS_NOT_FOUND,
    "commanders": WHEN_COMMANDERS_NOT_FOUND,
    "company": WHEN_COMPANIES_NOT_FOUND,
    "companies": WHEN_COMPANIES_NOT_FOUND,
    "platoon": WHEN_PLATOONS_NOT_FOUND,
    "platoons": WHEN_PLATOONS_NOT_FOUND,
    "user": WHEN_USERS_NOT_FOUND,
    "users": WHEN_USERS_NOT_FOUND,
    "batch": WHEN_BATCHES_NOT_FOUND,
    "batches": WHEN_BATCHES_NOT_FOUND,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_not_understood_message(use_alternative: bool = False) -> str:
    """Return the proper message when a user's question/answer is not understood."""
    if use_alternative:
        return random.choice(QUERY_PROCESSING_ISSUES["alternatives"])
    return QUERY_PROCESSING_ISSUES["main"]


def get_database_error_message(use_alternative: bool = False) -> str:
    """Return friendly message when database connection / system issues occur."""
    if use_alternative:
        return random.choice(DATABASE_CONNECTION_ISSUES["alternatives"])
    return DATABASE_CONNECTION_ISSUES["main"]


def get_entity_not_found_message(entity_name: Optional[str] = None, use_alternative: bool = False) -> str:
    """Return specific not found message based on entity type (Agniveer, Commander, Company, Platoon, User, Batch, or General)."""
    if not entity_name:
        target = WHEN_CANNOT_FIND_WHAT_LOOKING_FOR
    else:
        key = entity_name.strip().lower()
        target = _ENTITY_NOT_FOUND_MAP.get(key, WHEN_CANNOT_FIND_WHAT_LOOKING_FOR)

    if use_alternative:
        return random.choice(target["alternatives"])
    return target["main"]


def get_specific_record_not_found_message(use_alternative: bool = False) -> str:
    """Return message when a specific requested record cannot be located."""
    if use_alternative:
        return random.choice(WHEN_SPECIFIC_RECORD_NOT_FOUND["alternatives"])
    return WHEN_SPECIFIC_RECORD_NOT_FOUND["main"]
