"""
question_bank.py
================
Curated real question bank for AgniAI, parsed from the operation-level
test suite (13 categories x 47 operations x 4 query types) plus the
expanded cross-filter / multi-independent / comparison suite.

QUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]
QUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)
"""

QUESTION_BANK = {'by_category': {},
 'mixed': {'compare': ['Compare BPET and PPT scores.',
                       'Firing vs drill grading distribution.',
                       'BPET versus firing average marks.',
                       'Difference between PPT and drill performance.',
                       'Compare best attempts in BPET and PPT.',
                       'BPET vs drill improvement trends.',
                       'Compare top performers of Lakhwinder and Jaswant company in '
                       'BPET.',
                       'Arora vs Thorat company attendance this month.',
                       'Compare leave taken by Lak and Jas company.',
                       'BMI distribution of Lakhwinder vs Arora company.',
                       'Compare equipment stats of Jaswant and Thorat company.',
                       'Pending verifications of Lak vs Arora company.',
                       'Compare strength of Lakhwinder and Thorat company.',
                       'Cricket players in Jas vs Arora company.',
                       'Compare disqualified agniveers of Lakhwinder and Jaswant '
                       'company.',
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
                       'Compare monthly attendance of April and May for Lakhwinder '
                       'company.',
                       'Disqualifications in 2025 versus 2026.',
                       'Compare equipment returns in May and June.',
                       'This week vs last week attendance.',
                       'Compare attempt 1 and attempt 2 scores in BPET.',
                       'Attempt 2 vs attempt 3 performance in firing.',
                       'Compare first attempt and second attempt marks in PPT.',
                       'Improvement from attempt 1 to attempt 3 in drill.',
                       'Compare BPET scores of agniveer A0701516F and agniveer '
                       'A0701518M.',
                       'Agniveer A0701520K vs agniveer A0701522P overall performance.',
                       'Compare medical records of agniveer A0701523X and agniveer '
                       'A0701524A.',
                       'Difference between leave taken by agniveer A0701527N and '
                       'A0701528W.',
                       'Agniveer A0701530N versus agniveer A0701531W attendance.',
                       'Compare equipment issued to agniveer A0701536P and agniveer '
                       'A0701537X.',
                       'Compare BPET scores of cricket players in Lakhwinder and '
                       'Jaswant company.',
                       "Volleyball vs football players' attendance this month.",
                       'Compare leave taken by Sikh class agniveers of Arora and '
                       'Thorat company.',
                       'Overweight agniveers in Lak vs Jas company.',
                       'Compare top performers of platoon 1 and platoon 2 who play '
                       'hockey.',
                       'Combat Coat holders in Lakhwinder vs Thorat company.',
                       'Compare BPET improvement of batch 3 and batch 4 cricket '
                       'players.',
                       'Pending verifications of football players in Arora vs Jaswant '
                       'company.'],
           'cross_filter': ['Top performers in BPET who also play volleyball.',
                            'Highest scorers in PPT who play cricket.',
                            'Bottom performers in firing who also play football.',
                            'Who got excellent grade in drill among basketball '
                            'players?',
                            'Best attempt scores in BPET for agniveers who play '
                            'kabaddi.',
                            'Average PPT score of hockey players.',
                            'Who improved in BPET among the badminton players?',
                            'Top 10 in drill who also play tennis.',
                            'Whose scores dropped in PPT among the swimming team?',
                            'Grading summary of BPET for cricket players.',
                            'Performance trend in firing for football players.',
                            'Top performers in BPET from Sikh class.',
                            'Who got excellent in firing among Dogra class agniveers?',
                            'Overweight agniveers who are bottom performers in BPET.',
                            'Top performers in PPT who are underweight.',
                            'Who got excellent in firing among agniveers with normal '
                            'BMI?',
                            'Whose scores dropped in drill among agniveers with blood '
                            'group O+?',
                            'Top performers in firing who are physically fit by BMI.',
                            'Which agniveers suffering from dengue are still marked '
                            'present today?',
                            'Top performers in BPET who are currently on leave.',
                            'Who among the highest PPT scorers is absent today?',
                            'Bottom performers in firing who took the most leave.',
                            'Who got excellent in drill among agniveers currently on '
                            'medical leave?',
                            'Top performers in firing who never went on leave.',
                            'Top performers in BPET who are holding overdue equipment.',
                            'Combat Coat holders and their PPT scores.',
                            'Bottom performers in firing who returned damaged items.',
                            'Who got excellent in BPET among agniveers still holding '
                            'issued items?',
                            'Top performers in BPET who are present today.',
                            'Highest PPT scorers with pending verification.',
                            'Bottom performers in firing who are fully verified.',
                            'Top performers in PPT whose verification is not yet '
                            'responded.',
                            'Which cricket players are currently on leave?',
                            'Football players who took the most leave.',
                            'Hockey players who went AWOL.',
                            'Which badminton players are away right now?',
                            'Cricket players who are overweight.',
                            'Football players with blood group O+.',
                            'Underweight agniveers who play hockey.',
                            'Cricket players holding overdue equipment.',
                            'Football players who returned items in poor condition.',
                            'Hockey players with pending verification.',
                            'Combat Coat holders who play cricket.',
                            'Who took the most leave among agniveers with dengue?',
                            'Blood group of agniveers who went AWOL.',
                            'Overweight agniveers who are holding overdue equipment.',
                            'Agniveers with pending verification who are also absent '
                            'today.',
                            'Top leave takers who are still holding overdue equipment.',
                            'Absconded agniveers who never returned their Kit Bag.',
                            'Who took the most leave among fully verified agniveers?',
                            'AWOL agniveers who are still holding a Combat Coat.',
                            'Disqualified agniveers who were top BPET performers.',
                            'Removed agniveers who were still holding equipment.',
                            'Disqualified agniveers who had medical issues.',
                            'Personal details of the BPET top performers.',
                            'Profiles of agniveers who are currently on leave.',
                            'Contact info of agniveers with pending verification.',
                            'Personal details of overweight agniveers.',
                            'Top performers in BPET from Lakhwinder company who also '
                            'play volleyball.',
                            'Cricket players in Jaswant company who are currently on '
                            'leave.',
                            'Football players from platoon 3 holding overdue '
                            'equipment.',
                            'Top leave takers in Lakhwinder company who got excellent '
                            'in firing.',
                            'Underweight agniveers in Jaswant company who are absent '
                            'today.'],
           'multi_independent': ['Show top performers in BPET and also who plays '
                                 'volleyball.',
                                 "Who is on leave today and what's the BMI "
                                 'distribution?',
                                 'Give me the equipment stats and the pending '
                                 'verifications.',
                                 "Cricket players list and today's training schedule.",
                                 'Average PPT score and the blood group distribution.',
                                 'Who is present today and who is holding overdue '
                                 'equipment?',
                                 'Strength breakdown and the top leave takers.',
                                 'Disqualified agniveers and the absconded agniveers.',
                                 'Grading summary of BPET and the attendance summary '
                                 'for this month.',
                                 'Latest distribution and who got excellent in firing.',
                                 'Medical record of agniveer A0701516F and his '
                                 'equipment list.',
                                 'Personal details of agniveer A0701544N and his leave '
                                 'history.',
                                 'Who improved in drill and who is sick with fever?',
                                 'Football players and the rejected verification '
                                 'cases.',
                                 "Today's schedule and the unassigned agniveers.",
                                 'BPET trend and the disease statistics.',
                                 'Who topped PPT and how many are present today in '
                                 'Lakhwinder company?',
                                 'Kit Bag holders and the Sikh class roster.',
                                 'Attendance summary of Jaswant company and its '
                                 'strength breakdown.',
                                 'Best attempts in firing and the AWOL cases.',
                                 'Show top performers in BPET, who is on leave today, '
                                 'and the equipment stats.',
                                 "Cricket players, the BMI distribution, and today's "
                                 'schedule.',
                                 'Attendance summary, pending verifications, and the '
                                 'absconded list.',
                                 'Average PPT score, blood group distribution, and the '
                                 'strength breakdown.',
                                 'Who is present today, who plays hockey, and who is '
                                 'holding overdue equipment?',
                                 'Grading summary of firing, top leave takers, and the '
                                 'disqualified agniveers.',
                                 'Latest distribution, the unassigned agniveers, and '
                                 "today's training plan.",
                                 'Who improved in BPET, the disease statistics, and '
                                 'the rejected verifications.',
                                 'Strength of Lakhwinder company, its attendance '
                                 'summary, and its equipment stats.']}}
