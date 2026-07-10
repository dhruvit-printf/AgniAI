import pprint
import sys
sys.path.insert(0, 'e:/AgniAI')
from question_bank import QUESTION_BANK

new_questions = [
    "Which Agniveers whose police verification is pending are currently on leave?",
    "Which Agniveers who scored Excellent in BPET have completed police verification?",
    "Show Agniveers who failed Firing and still have issued equipment.",
    "Which Agniveers who are absent today have pending police verification?",
    "Find Agniveers who are medically unfit and currently on leave.",
    "Which BPET toppers are still holding issued equipment?",
    "Show Agniveers who completed police verification but haven't returned their equipment.",
    "Which Agniveers with rejected verification attended today's training?",
    "Find Agniveers who crossed the leave threshold and are medically unfit.",
    "Which Excellent performers are currently on leave?",
    "Show Agniveers who have Normal BMI and completed police verification.",
    "Which Agniveers who are present today still have issued equipment?",
    "Find BPET failures whose verification is completed.",
    "Which Agniveers with pending verification are holding Combat Dress?",
    "Show Agniveers who are overweight and scored Excellent in BPET.",
    "Which Agniveers who attended today's training still have pending verification?",
    "Find Agniveers who returned equipment and completed police verification.",
    "Which Agniveers who are on leave still have issued equipment?",
    "Show Excellent performers who are absent today.",
    "Which Agniveers with rejected verification are medically fit?",
    "Find Agniveers who failed Drill and are currently on leave.",
    "Which verified Agniveers are still holding issued equipment?",
    "Show Agniveers who are underweight and completed police verification.",
    "Which BPET toppers are absent today?",
    "Find Agniveers who haven't returned equipment and have pending verification.",
    "Which Agniveers diagnosed with fever are currently on leave?",
    "Show Agniveers who completed verification and attended today's parade.",
    "Which Excellent performers are medically unfit?",
    "Find Agniveers who failed Firing and have pending verification.",
    "Which Agniveers who are present today completed police verification?",
    "Show Agniveers who are currently absent and still holding equipment.",
    "Which Agniveers whose verification is rejected have returned their equipment?",
    "Find BPET toppers who are medically fit.",
    "Which Agniveers on leave have completed police verification?",
    "Show Agniveers who are hospitalized and have pending verification.",
    "Which Excellent performers haven't returned their issued kit?",
    "Find Agniveers who are overweight and currently on leave.",
    "Which Agniveers with completed verification are present today?",
    "Show BPET failures who still have issued equipment.",
    "Which Agniveers who attended training today have Normal BMI?",
    "Find Agniveers who are medically unfit and holding issued equipment.",
    "Which Agniveers with pending verification are absent today?",
    "Show Excellent performers who completed police verification.",
    "Which Agniveers who failed BPET are medically unfit?",
    "Find Agniveers who returned equipment and are currently present.",
    "Which Agniveers whose verification is completed are on leave?",
    "Show Agniveers who scored Good in Firing and completed police verification.",
    "Which Agniveers with rejected verification still have issued equipment?",
    "Find Agniveers who attended today's training and haven't returned equipment.",
    "Which Agniveers who are medically fit still have pending police verification?"
]

QUESTION_BANK['mixed']['cross_filter'] = new_questions

with open('e:/AgniAI/question_bank.py', 'w', encoding='utf-8') as f:
    f.write('"""\nquestion_bank.py\n================\nCurated real question bank for AgniAI, parsed from the operation-level\ntest suite (13 categories x 47 operations x 4 query types) plus the\nexpanded cross-filter / multi-independent / comparison suite.\n\nQUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]\nQUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)\n"""\n\n')
    f.write('QUESTION_BANK = \\\n')
    f.write(pprint.pformat(QUESTION_BANK, width=100))
    f.write('\n')
