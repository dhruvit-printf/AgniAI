"""
question_bank.py
================
Curated real question bank for AgniAI, parsed from the operation-level
test suite (13 categories x 47 operations x 4 query types) plus the
expanded cross-filter / multi-independent / comparison suite.

QUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]
QUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)
"""

QUESTION_BANK = \
{'by_category': {},
 'mixed': {'compare': ['Compare BPET and PPT scores.',
                       'Firing vs drill grading distribution.',
                       'BPET versus firing average marks.',
                       'Difference between PPT and drill performance.',
                       'Compare best attempts in BPET and PPT.',
                       'BPET vs drill improvement trends.',
                       'Compare top performers of Lakhwinder and Jaswant company in BPET.',
                       'Arora vs Thorat company attendance this month.',
                       'Compare leave taken by Lak and Jas company.',
                       'BMI distribution of Lakhwinder vs Arora company.',
                       'Compare equipment stats of Jaswant and Thorat company.',
                       'Pending verifications of Lak vs Arora company.',
                       'Compare strength of Lakhwinder and Thorat company.',
                       'Cricket players in Jas vs Arora company.',
                       'Compare disqualified agniveers of Lakhwinder and Jaswant company.',
                       'Overall performance of Arora versus Thorat company.',
                       'Compare BPET scores of platoon 1 and platoon 2.',
                       'Batch 3 vs batch 4 attendance.',
                       'Compare leave records of platoon 2 and platoon 5.',
                       'Blood group distribution of batch 1 vs batch 2.',
                       'Compare overdue equipment of platoon 3 and platoon 6.',
                       'Strength of batch 5 versus batch 6.',
                       'Compare cricket and football players.',
                       'Volleyball vs kabaddi participation.',
                       'Compare BPET scores of cricket and hockey players.',
                       'Sikh vs Dogra class strength.',
                       'Compare leave taken by football and basketball players.',
                       'BMI of cricket vs volleyball players.',
                       'Compare attendance of hockey and badminton players.',
                       'Compare attendance in May and June.',
                       'Leave taken in 2025 vs 2026.',
                       'Compare monthly attendance of April and May for Lakhwinder company.',
                       'Disqualifications in 2025 versus 2026.',
                       'Compare equipment returns in May and June.',
                       'This week vs last week attendance.',
                       'Compare attempt 1 and attempt 2 scores in BPET.',
                       'Attempt 2 vs attempt 3 performance in firing.',
                       'Compare first attempt and second attempt marks in PPT.',
                       'Improvement from attempt 1 to attempt 3 in drill.',
                       'Compare BPET scores of agniveer A0701516F and agniveer A0701518M.',
                       'Agniveer A0701520K vs agniveer A0701522P overall performance.',
                       'Compare medical records of agniveer A0701523X and agniveer A0701524A.',
                       'Difference between leave taken by agniveer A0701527N and A0701528W.',
                       'Agniveer A0701530N versus agniveer A0701531W attendance.',
                       'Compare equipment issued to agniveer A0701536P and agniveer A0701537X.',
                       'Compare BPET scores of cricket players in Lakhwinder and Jaswant company.',
                       "Volleyball vs football players' attendance this month.",
                       'Compare leave taken by Sikh class agniveers of Arora and Thorat company.',
                       'Overweight agniveers in Lak vs Jas company.',
                       'Compare top performers of platoon 1 and platoon 2 who play hockey.',
                       'Combat Coat holders in Lakhwinder vs Thorat company.',
                       'Compare BPET improvement of batch 3 and batch 4 cricket players.',
                       'Pending verifications of football players in Arora vs Jaswant company.'],
           'cross_filter': ['Which Agniveers whose police verification is pending are currently on '
                            'leave?',
                            'Which Agniveers who scored Excellent in BPET have completed police '
                            'verification?',
                            'Show Agniveers who failed Firing and still have issued equipment.',
                            'Which Agniveers who are absent today have pending police '
                            'verification?',
                            'Find Agniveers who are medically unfit and currently on leave.',
                            'Which BPET toppers are still holding issued equipment?',
                            "Show Agniveers who completed police verification but haven't returned "
                            'their equipment.',
                            "Which Agniveers with rejected verification attended today's training?",
                            'Find Agniveers who crossed the leave threshold and are medically '
                            'unfit.',
                            'Which Excellent performers are currently on leave?',
                            'Show Agniveers who have Normal BMI and completed police verification.',
                            'Which Agniveers who are present today still have issued equipment?',
                            'Find BPET failures whose verification is completed.',
                            'Which Agniveers with pending verification are holding Combat Dress?',
                            'Show Agniveers who are overweight and scored Excellent in BPET.',
                            "Which Agniveers who attended today's training still have pending "
                            'verification?',
                            'Find Agniveers who returned equipment and completed police '
                            'verification.',
                            'Which Agniveers who are on leave still have issued equipment?',
                            'Show Excellent performers who are absent today.',
                            'Which Agniveers with rejected verification are medically fit?',
                            'Find Agniveers who failed Drill and are currently on leave.',
                            'Which verified Agniveers are still holding issued equipment?',
                            'Show Agniveers who are underweight and completed police verification.',
                            'Which BPET toppers are absent today?',
                            "Find Agniveers who haven't returned equipment and have pending "
                            'verification.',
                            'Which Agniveers diagnosed with fever are currently on leave?',
                            "Show Agniveers who completed verification and attended today's "
                            'parade.',
                            'Which Excellent performers are medically unfit?',
                            'Find Agniveers who failed Firing and have pending verification.',
                            'Which Agniveers who are present today completed police verification?',
                            'Show Agniveers who are currently absent and still holding equipment.',
                            'Which Agniveers whose verification is rejected have returned their '
                            'equipment?',
                            'Find BPET toppers who are medically fit.',
                            'Which Agniveers on leave have completed police verification?',
                            'Show Agniveers who are hospitalized and have pending verification.',
                            "Which Excellent performers haven't returned their issued kit?",
                            'Find Agniveers who are overweight and currently on leave.',
                            'Which Agniveers with completed verification are present today?',
                            'Show BPET failures who still have issued equipment.',
                            'Which Agniveers who attended training today have Normal BMI?',
                            'Find Agniveers who are medically unfit and holding issued equipment.',
                            'Which Agniveers with pending verification are absent today?',
                            'Show Excellent performers who completed police verification.',
                            'Which Agniveers who failed BPET are medically unfit?',
                            'Find Agniveers who returned equipment and are currently present.',
                            'Which Agniveers whose verification is completed are on leave?',
                            'Show Agniveers who scored Good in Firing and completed police '
                            'verification.',
                            'Which Agniveers with rejected verification still have issued '
                            'equipment?',
                            "Find Agniveers who attended today's training and haven't returned "
                            'equipment.',
                            'Which Agniveers who are medically fit still have pending police '
                            'verification?'],
           'multi_independent': ['Show top performers in BPET and also who plays volleyball.',
                                 "Who is on leave today and what's the BMI distribution?",
                                 'Give me the equipment stats and the pending verifications.',
                                 "Cricket players list and today's training schedule.",
                                 'Average PPT score and the blood group distribution.',
                                 'Who is present today and who is holding overdue equipment?',
                                 'Strength breakdown and the top leave takers.',
                                 'Disqualified agniveers and the absconded agniveers.',
                                 'Grading summary of BPET and the attendance summary for this '
                                 'month.',
                                 'Latest distribution and who got excellent in firing.',
                                 'Medical record of agniveer A0701516F and his equipment list.',
                                 'Personal details of agniveer A0701544N and his leave history.',
                                 'Who improved in drill and who is sick with fever?',
                                 'Football players and the rejected verification cases.',
                                 "Today's schedule and the unassigned agniveers.",
                                 'BPET trend and the disease statistics.',
                                 'Who topped PPT and how many are present today in Lakhwinder '
                                 'company?',
                                 'Kit Bag holders and the Sikh class roster.',
                                 'Attendance summary of Jaswant company and its strength '
                                 'breakdown.',
                                 'Best attempts in firing and the AWOL cases.',
                                 'Show top performers in BPET, who is on leave today, and the '
                                 'equipment stats.',
                                 "Cricket players, the BMI distribution, and today's schedule.",
                                 'Attendance summary, pending verifications, and the absconded '
                                 'list.',
                                 'Average PPT score, blood group distribution, and the strength '
                                 'breakdown.',
                                 'Who is present today, who plays hockey, and who is holding '
                                 'overdue equipment?',
                                 'Grading summary of firing, top leave takers, and the '
                                 'disqualified agniveers.',
                                 "Latest distribution, the unassigned agniveers, and today's "
                                 'training plan.',
                                 'Who improved in BPET, the disease statistics, and the rejected '
                                 'verifications.',
                                 'Strength of Lakhwinder company, its attendance summary, and its '
                                 'equipment stats.']}}
