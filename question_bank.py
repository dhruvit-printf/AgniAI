"""
question_bank.py
================
Curated real question bank for AgniAI, parsed from the operation-level
test suite (13 categories x 47 operations x 4 query types) plus the
expanded cross-filter / multi-independent / comparison suite.

QUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]
QUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)

NOTE: "cross_filter" questions are intentionally kept to only those where
two conditions apply to the SAME subject and are evaluated jointly (a
true intersection filter), as opposed to "multi_independent" questions
which ask for two separate, unrelated result sets. Each cross_filter
question is phrased with a contrastive/consequential connector ("but
still", "yet", "despite", "and still") rather than a bare "and", so the
second condition clearly qualifies the first on the same entity instead
of reading as a second, unrelated fact loosely appended to the first.
"""

QUESTION_BANK = {
    "by_category": {},
    "mixed": {
        "compare": [
            "Compare BPET and PPT scores.",
            "Firing vs drill grading distribution.",
            "BPET versus firing average marks.",
            "Difference between PPT and drill performance.",
            "Compare best attempts in BPET and PPT.",
            "BPET vs drill improvement trends.",
            "Compare top performers of Lakhwinder and Jaswant company in BPET.",
            "Arora vs Thorat company attendance this month.",
            "Compare leave taken by Lak and Jas company.",
            "BMI distribution of Lakhwinder vs Arora company.",
            "Compare equipment stats of Jaswant and Thorat company.",
            "Pending verifications of Lak vs Arora company.",
            "Compare strength of Lakhwinder and Thorat company.",
            "Cricket players in Jas vs Arora company.",
            "Compare disqualified agniveers of Lakhwinder and Jaswant company.",
            "Overall performance of Arora versus Thorat company.",
            "Compare BPET scores of platoon 1 and platoon 2.",
            "Batch 3 vs batch 4 attendance.",
            "Compare leave records of platoon 2 and platoon 5.",
            "Blood group distribution of batch 1 vs batch 2.",
            "Compare overdue equipment of platoon 3 and platoon 6.",
            "Strength of batch 5 versus batch 6.",
            "Compare cricket and football players.",
            "Volleyball vs kabaddi participation.",
            "Compare BPET scores of cricket and hockey players.",
            "Sikh vs Dogra class strength.",
            "Compare leave taken by football and basketball players.",
            "BMI of cricket vs volleyball players.",
            "Compare attendance of hockey and badminton players.",
            "Compare attendance in May and June.",
            "Leave taken in 2025 vs 2026.",
            "Compare monthly attendance of April and May for Lakhwinder company.",
            "Disqualifications in 2025 versus 2026.",
            "Compare equipment returns in May and June.",
            "This week vs last week attendance.",
            "Compare attempt 1 and attempt 2 scores in BPET.",
            "Attempt 2 vs attempt 3 performance in firing.",
            "Compare first attempt and second attempt marks in PPT.",
            "Improvement from attempt 1 to attempt 3 in drill.",
            "Compare BPET scores of agniveer A0701516F and agniveer A0701518M.",
            "Agniveer A0701520K vs agniveer A0701522P overall performance.",
            "Compare medical records of agniveer A0701523X and agniveer A0701524A.",
            "Difference between leave taken by agniveer A0701527N and A0701528W.",
            "Agniveer A0701530N versus agniveer A0701531W attendance.",
            "Compare equipment issued to agniveer A0701536P and agniveer A0701537X.",
            "Compare BPET scores of cricket players in Lakhwinder and Jaswant company.",
            "Volleyball vs football players' attendance this month.",
            "Compare leave taken by Sikh class agniveers of Arora and Thorat company.",
            "Overweight agniveers in Lak vs Jas company.",
            "Compare top performers of platoon 1 and platoon 2 who play hockey.",
            "Combat Coat holders in Lakhwinder vs Thorat company.",
            "Compare BPET improvement of batch 3 and batch 4 cricket players.",
            "Pending verifications of football players in Arora vs Jaswant company.",
        ],
        "cross_filter": [
            "Show Agniveers who failed Firing but still have issued equipment.",
            "Find Agniveers who are medically unfit yet still marked on leave.",
            "Find Agniveers who crossed the leave threshold and are still "
            "medically unfit.",
            "Show Agniveers who have Normal BMI but still have pending police "
            "verification.",
            "Show Agniveers who are overweight despite scoring Excellent in BPET.",
            "Find Agniveers who returned equipment but still have pending police "
            "verification.",
            "Find Agniveers who failed Drill and are, on top of that, currently "
            "on leave.",
            "Show Agniveers who are underweight despite completing police "
            "verification.",
            "Find Agniveers who haven't returned equipment and still have "
            "verification pending too.",
            "Show Agniveers who completed verification but still missed today's "
            "parade.",
            "Find Agniveers who failed Firing and still have verification " "pending.",
            "Show Agniveers who are currently absent yet still holding " "equipment.",
            "Show Agniveers who are hospitalized and still awaiting police "
            "verification.",
            "Find Agniveers who are overweight yet still currently on leave.",
            "Find Agniveers who are medically unfit but still holding issued "
            "equipment.",
            "Find Agniveers who returned equipment and are still marked "
            "present today.",
            "Show Agniveers who scored Good in Firing but still have police "
            "verification pending.",
            "Find Agniveers who attended today's training yet still haven't "
            "returned equipment.",
        ],
        "multi_independent": [
            "Show top performers in BPET and also who plays volleyball.",
            "Who is on leave today and what's the BMI distribution?",
            "Give me the equipment stats and the pending verifications.",
            "Cricket players list and today's training schedule.",
            "Average PPT score and the blood group distribution.",
            "Who is present today and who is holding overdue equipment?",
            "Strength breakdown and the top leave takers.",
            "Disqualified agniveers and the absconded agniveers.",
            "Grading summary of BPET and the attendance summary for this month.",
            "Latest distribution and who got excellent in firing.",
            "Medical record of agniveer A0701516F and his equipment list.",
            "Personal details of agniveer A0701544N and his leave history.",
            "Who improved in drill and who is sick with fever?",
            "Football players and the rejected verification cases.",
            "Today's schedule and the unassigned agniveers.",
            "BPET trend and the disease statistics.",
            "Who topped PPT and how many are present today in Lakhwinder company?",
            "Kit Bag holders and the Sikh class roster.",
            "Attendance summary of Jaswant company and its strength breakdown.",
            "Best attempts in firing and the AWOL cases.",
            "Show top performers in BPET, who is on leave today, and the "
            "equipment stats.",
            "Cricket players, the BMI distribution, and today's schedule.",
            "Attendance summary, pending verifications, and the absconded list.",
            "Average PPT score, blood group distribution, and the strength breakdown.",
            "Who is present today, who plays hockey, and who is holding "
            "overdue equipment?",
            "Grading summary of firing, top leave takers, and the "
            "disqualified agniveers.",
            "Latest distribution, the unassigned agniveers, and today's training plan.",
            "Who improved in BPET, the disease statistics, and the rejected "
            "verifications.",
            "Strength of Lakhwinder company, its attendance summary, and its "
            "equipment stats.",
        ],
    },
}
