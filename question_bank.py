"""
question_bank.py
================
Curated real question bank for AgniAI, parsed from the operation-level
test suite (13 categories x 47 operations x 4 query types) plus the
expanded cross-filter / multi-independent / comparison suite.

QUESTION_BANK["by_category"][CATEGORY][SUBCATEGORY][QUERY_TYPE] -> list[str]
QUESTION_BANK["mixed"][QUERY_TYPE] -> list[str]  (category-spanning examples)
"""

QUESTION_BANK = {'by_category': {'ATTENDANCE': {'Daily': {'compare': ['Compare daily attendance of '
                                                      'Lakhwinder and Arora company.',
                                                      'Daily attendance on 2026-06-10 '
                                                      'vs 2026-06-15.',
                                                      'Compare day wise attendance of '
                                                      'platoon 3 and platoon 6.'],
                                          'cross_filter': ['Which agniveers were '
                                                           'marked present today '
                                                           'despite being on medical '
                                                           'leave?',
                                                           'Daily attendance of '
                                                           'agniveers who are also top '
                                                           'BPET performers.',
                                                           "Today's attendance for "
                                                           'agniveers who are '
                                                           'currently holding overdue '
                                                           'equipment.'],
                                          'multi_independent': ['Daily attendance and '
                                                                "today's schedule.",
                                                                'Day wise attendance '
                                                                'and the pending '
                                                                'verifications.',
                                                                'Daily attendance and '
                                                                'the BMI '
                                                                'distribution.'],
                                          'simple': ['Show daily attendance for today.',
                                                     'Give me the day wise attendance.',
                                                     'Attendance for 15/06/2026.',
                                                     'Daily attendance breakdown.']},
                                'Monthly': {'compare': ['Compare attendance in May and '
                                                        'June.',
                                                        'Monthly attendance of Lak vs '
                                                        'Jas company.',
                                                        'Compare attendance in 2025 '
                                                        'and 2026.'],
                                            'cross_filter': ['Monthly attendance of '
                                                             'agniveers who took the '
                                                             'most leave this year.',
                                                             "This month's attendance "
                                                             'for agniveers with '
                                                             'pending verification.',
                                                             'Monthly attendance of '
                                                             'cricket players.'],
                                            'multi_independent': ['Monthly attendance '
                                                                  'and the leave '
                                                                  'summary.',
                                                                  'Attendance for June '
                                                                  'and the blood group '
                                                                  'distribution.',
                                                                  "This month's "
                                                                  'attendance and the '
                                                                  'BPET average.'],
                                            'simple': ['Show monthly attendance for '
                                                       'June 2026.',
                                                       'What was the attendance last '
                                                       'month?',
                                                       'Give me the month wise '
                                                       'attendance report.',
                                                       'Attendance for the current '
                                                       'month.']},
                                'Present': {'compare': ['Compare present count of Lak '
                                                        'and Jas company.',
                                                        'Present today in platoon 2 vs '
                                                        'platoon 6.',
                                                        'Compare who is present in '
                                                        'batch 1 and batch 2.'],
                                            'cross_filter': ['Which agniveers are '
                                                             'present today despite '
                                                             'being on the '
                                                             'disqualified list?',
                                                             'Who is present today '
                                                             'from the agniveers '
                                                             'currently under medical '
                                                             'treatment?',
                                                             'Present count today for '
                                                             'agniveers who are '
                                                             'holding overdue '
                                                             'equipment.'],
                                            'multi_independent': ['Who is present '
                                                                  'today and who is on '
                                                                  'leave?',
                                                                  'Present count today '
                                                                  "and today's "
                                                                  'schedule.',
                                                                  'Who is here today '
                                                                  'and the equipment '
                                                                  'stats.'],
                                            'simple': ['How many agniveers are present '
                                                       'today?',
                                                       'Who is present on campus right '
                                                       'now?',
                                                       'Who came today?',
                                                       'Show me who is marked '
                                                       'present.']},
                                'Summary': {'compare': ['Compare attendance summary of '
                                                        'Arora and Thorat company.',
                                                        'Attendance overview of batch '
                                                        '1 vs batch 2.',
                                                        'Compare overall attendance of '
                                                        'platoon 1 and platoon 5.'],
                                            'cross_filter': ['Attendance summary for '
                                                             'agniveers who took the '
                                                             'most leave this year.',
                                                             'Attendance overview for '
                                                             'agniveers who are '
                                                             'currently disqualified.',
                                                             'Attendance summary of '
                                                             'football players.'],
                                            'multi_independent': ['Attendance summary '
                                                                  'and the BPET '
                                                                  'average score.',
                                                                  'Attendance overview '
                                                                  'and the '
                                                                  'disqualified list.',
                                                                  'Attendance recap '
                                                                  'and the latest '
                                                                  'distribution.'],
                                            'simple': ['Give me the attendance '
                                                       'summary.',
                                                       'Show the overall attendance '
                                                       'overview.',
                                                       'Attendance recap for the '
                                                       'batch.',
                                                       'General attendance snapshot.']},
                                'Weekly': {'compare': ['Compare this week and last '
                                                       'week attendance.',
                                                       'Weekly attendance of Jaswant '
                                                       'vs Thorat company.',
                                                       'Compare weekly attendance of '
                                                       'platoon 1 and platoon 2.'],
                                           'cross_filter': ['Weekly attendance of '
                                                            'agniveers who are '
                                                            'currently underweight.',
                                                            "This week's attendance "
                                                            'for agniveers still '
                                                            'holding a Combat Coat.',
                                                            'Weekly attendance of '
                                                            'Dogra class agniveers.'],
                                           'multi_independent': ['Weekly attendance '
                                                                 'and the equipment '
                                                                 'stats.',
                                                                 "This week's "
                                                                 'attendance and who '
                                                                 'is on leave.',
                                                                 'Week wise attendance '
                                                                 'and the strength '
                                                                 'breakdown.'],
                                           'simple': ['Show attendance for this week.',
                                                      'Give me the weekly attendance '
                                                      'report.',
                                                      "What was last week's "
                                                      'attendance?',
                                                      'Week wise attendance '
                                                      'breakdown.']}},
                 'DISQUALIFIED': {'removed': {'compare': ['Compare disqualified '
                                                          'agniveers of Lakhwinder and '
                                                          'Arora company.',
                                                          'Disqualifications in batch '
                                                          '1 vs batch 2.',
                                                          'Compare removed agniveers '
                                                          'in 2025 and 2026.'],
                                              'cross_filter': ['Disqualified agniveers '
                                                               'who were previously '
                                                               'top BPET performers.',
                                                               'Which disqualified '
                                                               'agniveers still have '
                                                               'equipment pending '
                                                               'return?',
                                                               'Disqualified agniveers '
                                                               'who had gone AWOL '
                                                               'before removal.'],
                                              'multi_independent': ['Disqualified '
                                                                    'agniveers and the '
                                                                    'absconded list.',
                                                                    'Removed agniveers '
                                                                    'and the pending '
                                                                    'verifications.',
                                                                    'Expelled list and '
                                                                    "today's "
                                                                    'schedule.'],
                                              'simple': ['List disqualified agniveers.',
                                                         'Who was removed from the '
                                                         'program?',
                                                         'Which agniveers were '
                                                         'expelled?',
                                                         'Show disqualified agniveers '
                                                         'with reasons.']}},
                 'DISTRIBUTION': {'ByUnit': {'compare': ['Compare unit wise '
                                                         'distribution of batch 1 and '
                                                         'batch 2.',
                                                         'Distribution by unit for '
                                                         'Arora vs Thorat company.',
                                                         'Compare unit wise breakdown '
                                                         'of platoon 1 and platoon 3.'],
                                             'cross_filter': ['Distribution by unit '
                                                              'for agniveers who play '
                                                              'cricket.',
                                                              'Unit wise distribution '
                                                              'of agniveers currently '
                                                              'on leave.',
                                                              'Distribution by unit '
                                                              'for Sikh class '
                                                              'agniveers.'],
                                             'multi_independent': ['Distribution by '
                                                                   'unit and the '
                                                                   'attendance '
                                                                   'summary.',
                                                                   'Unit wise '
                                                                   'breakdown and the '
                                                                   'pending '
                                                                   'verifications.',
                                                                   'Distribution '
                                                                   'across units and '
                                                                   'the BMI analysis.'],
                                             'simple': ['Show distribution by unit.',
                                                        'Give me the unit wise '
                                                        'breakdown of agniveers.',
                                                        'Agniveers grouped by unit.',
                                                        'How is the distribution '
                                                        'across units?']},
                                  'Latest': {'compare': ['Compare latest distributions '
                                                         'of Lak and Jas company.',
                                                         'Recent distributions in May '
                                                         'vs June.',
                                                         'Compare latest allocations '
                                                         'of batch 4 and batch 6.'],
                                             'cross_filter': ['Latest distribution for '
                                                              'agniveers who are top '
                                                              'BPET performers.',
                                                              'Recent distribution of '
                                                              'agniveers with pending '
                                                              'verification.',
                                                              'Latest allocation for '
                                                              'Dogra class agniveers.'],
                                             'multi_independent': ['Latest '
                                                                   'distribution and '
                                                                   "today's schedule.",
                                                                   'Recent allocations '
                                                                   'and the equipment '
                                                                   'stats.',
                                                                   'Newest '
                                                                   'distribution and '
                                                                   'the BPET top '
                                                                   'performers.'],
                                             'simple': ['Show the latest distribution.',
                                                        'Which agniveers were recently '
                                                        'distributed?',
                                                        'Show the most recent '
                                                        'allocations.',
                                                        'Give me the newest '
                                                        'distribution report.']},
                                  'TopUnit': {'compare': ['Compare top units for batch '
                                                          '1 and batch 2.',
                                                          'Top unit by distribution '
                                                          'for Lak vs Jas company.',
                                                          'Compare the leading unit in '
                                                          '2025 and 2026.'],
                                              'cross_filter': ['Which unit received '
                                                               'the most agniveers who '
                                                               'play football?',
                                                               'Top unit by '
                                                               'distribution among '
                                                               'fully verified '
                                                               'agniveers.',
                                                               'Which unit has the '
                                                               'most agniveers '
                                                               'currently on leave?'],
                                              'multi_independent': ['Top unit and the '
                                                                    'strength '
                                                                    'breakdown.',
                                                                    'Which unit has '
                                                                    'the most '
                                                                    'agniveers and '
                                                                    "today's schedule.",
                                                                    'Top unit by '
                                                                    'distribution and '
                                                                    'the equipment '
                                                                    'stats.'],
                                              'simple': ['Which unit received the most '
                                                         'agniveers?',
                                                         'Show me the top unit by '
                                                         'distribution.',
                                                         'Which unit has the highest '
                                                         'agniveer count?',
                                                         'What is the leading unit in '
                                                         'distribution?']},
                                  'Unassigned': {'compare': ['Compare unassigned '
                                                             'agniveers of Lakhwinder '
                                                             'and Jaswant company.',
                                                             'Unassigned counts in '
                                                             'batch 3 vs batch 4.',
                                                             'Compare pending '
                                                             'assignments in 2025 and '
                                                             '2026.'],
                                                 'cross_filter': ['Unassigned '
                                                                  'agniveers who are '
                                                                  'top performers in '
                                                                  'PPT.',
                                                                  'Which unassigned '
                                                                  'agniveers are '
                                                                  'disqualified?',
                                                                  'Unassigned '
                                                                  'agniveers who are '
                                                                  'currently holding '
                                                                  'issued equipment.'],
                                                 'multi_independent': ['Unassigned '
                                                                       'agniveers and '
                                                                       'the pending '
                                                                       'verifications.',
                                                                       'Pending '
                                                                       'assignments '
                                                                       'and the '
                                                                       'current leave '
                                                                       'status.',
                                                                       'Unassigned '
                                                                       'list and the '
                                                                       'disqualified '
                                                                       'agniveers.'],
                                                 'simple': ['Who is not yet assigned?',
                                                            'List unassigned '
                                                            'agniveers.',
                                                            'Which agniveers are '
                                                            'pending assignment?',
                                                            'Show agniveers without '
                                                            'any allocation.']}},
                 'EQUIPMENT': {'AgniveerWise': {'compare': ['Compare equipment of '
                                                            'agniveer A0701700M and '
                                                            'agniveer A0701701P.',
                                                            'Agniveer wise equipment '
                                                            'of Lakhwinder vs Jaswant '
                                                            'company.',
                                                            'Compare individual '
                                                            'equipment of platoon 1 '
                                                            'and platoon 5.'],
                                                'cross_filter': ['Equipment issued to '
                                                                 'agniveers who are '
                                                                 'currently on leave.',
                                                                 'Agniveer wise '
                                                                 'equipment for '
                                                                 'cricket players.',
                                                                 'Equipment held by '
                                                                 'agniveers with '
                                                                 'pending '
                                                                 'verification.'],
                                                'multi_independent': ['Agniveer wise '
                                                                      'equipment and '
                                                                      'the attendance '
                                                                      'summary.',
                                                                      'Equipment of '
                                                                      'agniveer '
                                                                      'A0701690N and '
                                                                      'his personal '
                                                                      'details.',
                                                                      'Individual '
                                                                      'equipment lists '
                                                                      'and the '
                                                                      'strength '
                                                                      'breakdown.'],
                                                'simple': ['Show equipment issued to '
                                                           'each agniveer.',
                                                           'What equipment does '
                                                           'agniveer A0701664M have?',
                                                           'Give me the agniveer wise '
                                                           'equipment list.',
                                                           'Who has what equipment?']},
                               'Holding': {'compare': ['Compare overdue equipment of '
                                                       'Lakhwinder and Thorat company.',
                                                       'Equipment holding in platoon 2 '
                                                       'vs platoon 3.',
                                                       'Compare issued items held by '
                                                       'batch 4 and batch 5.'],
                                           'cross_filter': ['Which agniveers are '
                                                            'holding overdue equipment '
                                                            'despite being on leave?',
                                                            'Overdue equipment held by '
                                                            'agniveers who were '
                                                            'disqualified.',
                                                            'Equipment held by '
                                                            'agniveers currently '
                                                            'absent from campus.'],
                                           'multi_independent': ['Overdue equipment '
                                                                 'and the current '
                                                                 'leave status.',
                                                                 'Who is holding '
                                                                 'issued items and '
                                                                 "today's schedule.",
                                                                 'Equipment in '
                                                                 'possession and the '
                                                                 'BMI distribution.'],
                                           'simple': ['Who is currently holding '
                                                      'overdue equipment?',
                                                      'Which agniveers are still '
                                                      'holding issued items?',
                                                      'Show me who has equipment in '
                                                      'possession.',
                                                      'List items that have not been '
                                                      'returned yet.']},
                               'Returned': {'compare': ['Compare damaged returns of '
                                                        'Arora and Thorat company.',
                                                        'Poor condition returns in '
                                                        'batch 1 vs batch 2.',
                                                        'Compare returned items in May '
                                                        'and June.'],
                                            'cross_filter': ['Which agniveers returned '
                                                             'equipment in poor '
                                                             'condition and are also '
                                                             'bottom BPET performers?',
                                                             'Damaged returns from '
                                                             'agniveers who went AWOL.',
                                                             'Poor condition returns '
                                                             'from agniveers currently '
                                                             'on medical leave.'],
                                            'multi_independent': ['Damaged returns and '
                                                                  'the pending '
                                                                  'verifications.',
                                                                  'Poor condition '
                                                                  "returns and today's "
                                                                  'attendance.',
                                                                  'Returned items and '
                                                                  'the disqualified '
                                                                  'agniveers.'],
                                            'simple': ['Which items were returned in '
                                                       'poor condition?',
                                                       'List damaged equipment that '
                                                       'was returned.',
                                                       'Show me broken items handed '
                                                       'back.',
                                                       'What faulty equipment was sent '
                                                       'back?']},
                               'Search': {'compare': ['Compare Combat Coat issues of '
                                                      'Lak and Jas company.',
                                                      'Kit Bag holders in platoon 1 vs '
                                                      'platoon 2.',
                                                      'Compare Blanket issues of batch '
                                                      '2 and batch 3.'],
                                          'cross_filter': ['Which agniveers currently '
                                                           'on leave are still holding '
                                                           'a Combat Coat?',
                                                           'Kit Bag holders who are '
                                                           'disqualified.',
                                                           'Blanket issued to '
                                                           'agniveers with pending '
                                                           'verification.'],
                                          'multi_independent': ['Search for Combat '
                                                                "Coat and show today's "
                                                                'schedule.',
                                                                'Find Kit Bag holders '
                                                                'and the current leave '
                                                                'status.',
                                                                'Look up Blanket in '
                                                                'equipment and the '
                                                                'BPET toppers.'],
                                          'simple': ['Find who has been issued a '
                                                     'Combat Coat.',
                                                     'Search for Kit Bag in equipment.',
                                                     'Look up Blanket in the '
                                                     'inventory.',
                                                     'Check equipment records for Pt '
                                                     'Shoes Brown.']},
                               'Stats': {'compare': ['Compare equipment stats of '
                                                     'Lakhwinder and Jaswant company.',
                                                     'Equipment stats in May vs June.',
                                                     'Compare inventory summary of '
                                                     'batch 2 and batch 3.'],
                                         'cross_filter': ['Equipment stats for '
                                                          'agniveers who are top BPET '
                                                          'performers.',
                                                          'Equipment summary for '
                                                          'agniveers currently on '
                                                          'leave.',
                                                          'Inventory stats for Sikh '
                                                          'class agniveers.'],
                                         'multi_independent': ['Equipment stats and '
                                                               "today's attendance.",
                                                               'Equipment summary and '
                                                               'the strength '
                                                               'breakdown.',
                                                               'Inventory overview and '
                                                               'the pending '
                                                               'verifications.'],
                                         'simple': ['Show the equipment stats.',
                                                    'Give me the equipment inventory '
                                                    'summary.',
                                                    'How many items are issued '
                                                    'overall?',
                                                    'Show the equipment count '
                                                    'overview.']}},
                 'LEAVE': {'Absconded': {'compare': ['Compare absconded cases of '
                                                     'Lakhwinder and Arora company.',
                                                     'AWOL cases in 2025 vs 2026.',
                                                     'Compare absconded agniveers of '
                                                     'batch 2 and batch 6.'],
                                         'cross_filter': ['Absconded agniveers who '
                                                          'were previously top BPET '
                                                          'performers.',
                                                          'AWOL cases among agniveers '
                                                          'still holding issued '
                                                          'equipment.',
                                                          'Absconded agniveers with '
                                                          'pending verification.'],
                                         'multi_independent': ['Absconded agniveers '
                                                               'and the pending '
                                                               'verifications.',
                                                               'AWOL cases and the '
                                                               'current strength.',
                                                               'Absconded list and '
                                                               "today's attendance."],
                                         'simple': ['List all absconded agniveers.',
                                                    'Who went AWOL?',
                                                    'Which agniveers have gone '
                                                    'missing?',
                                                    'Show me agniveers who did not '
                                                    'return from leave.']},
                           'Current': {'compare': ['Compare current leave status of '
                                                   'Lak and Jas company.',
                                                   'Who is on leave in platoon 1 vs '
                                                   'platoon 4?',
                                                   'Compare currently absent agniveers '
                                                   'of batch 2 and batch 3.'],
                                       'cross_filter': ['Which agniveers are on leave '
                                                        'today despite pending '
                                                        'verification?',
                                                        'Currently absent agniveers '
                                                        'who are top performers in '
                                                        'PPT.',
                                                        'Who from the disqualified '
                                                        'list is currently marked on '
                                                        'leave?'],
                                       'multi_independent': ['Who is on leave today '
                                                             "and what is today's "
                                                             'schedule?',
                                                             'Current leave status and '
                                                             'how many are present '
                                                             'today.',
                                                             'Who is currently absent '
                                                             'and the strength '
                                                             'breakdown.'],
                                       'simple': ['Who is currently on leave?',
                                                  'Who is absent today?',
                                                  "Show me today's leave status.",
                                                  'Which agniveers are away right '
                                                  'now?']},
                           'Least': {'compare': ['Compare least leave takers of Arora '
                                                 'and Thorat company.',
                                                 'Fewest leaves in platoon 2 vs '
                                                 'platoon 3.',
                                                 'Compare least leave taken in 2025 '
                                                 'and 2026.'],
                                     'cross_filter': ['Least leave taken among '
                                                      'agniveers who are top BPET '
                                                      'performers.',
                                                      'Fewest leaves among agniveers '
                                                      'currently holding overdue '
                                                      'equipment.',
                                                      'Most regular agniveers among '
                                                      'Sikh class.'],
                                     'multi_independent': ['Who took the least leave '
                                                           'and who topped BPET?',
                                                           'Fewest leave takers and '
                                                           'the attendance summary.',
                                                           'Least leave taken and the '
                                                           'pending verifications.'],
                                     'simple': ['Who took the least leave?',
                                                'Which agniveers have the fewest days '
                                                'absent?',
                                                'Who is most regular with the lowest '
                                                'leave count?',
                                                'Show agniveers who hardly took any '
                                                'leave.']},
                           'Most': {'compare': ['Compare leave taken by Lakhwinder and '
                                                'Jaswant company.',
                                                'Most leave taken in platoon 1 vs '
                                                'platoon 2.',
                                                'Compare top leave takers of batch 3 '
                                                'and batch 4.'],
                                    'cross_filter': ['Who took the most leave among '
                                                     'agniveers who are bottom BPET '
                                                     'performers?',
                                                     'Top leave takers among agniveers '
                                                     'with pending verification.',
                                                     'Most leave taken among agniveers '
                                                     'currently holding overdue '
                                                     'equipment.'],
                                    'multi_independent': ['Top leave takers and the '
                                                          'BPET average score.',
                                                          'Who took the most leave and '
                                                          "today's schedule.",
                                                          'Most leave taken and the '
                                                          'equipment stats.'],
                                    'simple': ['Who has taken the most leave?',
                                               'Show me the top leave takers.',
                                               'Who took maximum leave this year?',
                                               'Which agniveers have the highest leave '
                                               'count?']}},
                 'MEDICAL': {'BMI': {'compare': ['Compare BMI distribution of '
                                                 'Lakhwinder and Jaswant company.',
                                                 'Overweight counts in platoon 1 vs '
                                                 'platoon 3.',
                                                 'Compare BMI of batch 2 and batch 4.'],
                                     'cross_filter': ['Overweight agniveers who are '
                                                      'bottom performers in BPET.',
                                                      'Underweight agniveers currently '
                                                      'on leave.',
                                                      'BMI distribution of agniveers '
                                                      'with pending verification.'],
                                     'multi_independent': ['BMI distribution and '
                                                           "today's attendance.",
                                                           "Who is obese and today's "
                                                           'training schedule.',
                                                           'Overweight agniveers and '
                                                           'the BPET toppers.'],
                                     'simple': ['Show the BMI distribution.',
                                                'Who is overweight?',
                                                'Which agniveers are underweight?',
                                                'Who is the fittest by BMI?']},
                             'BloodGroup': {'compare': ['Compare blood group '
                                                        'distribution of Lak and Jas '
                                                        'company.',
                                                        'Blood groups in batch 1 vs '
                                                        'batch 3.',
                                                        'Compare O+ counts of Arora '
                                                        'and Thorat company.'],
                                            'cross_filter': ['Blood group of agniveers '
                                                             'currently on medical '
                                                             'leave.',
                                                             'O+ agniveers who are top '
                                                             'BPET performers.',
                                                             'Blood group distribution '
                                                             'of agniveers who went '
                                                             'AWOL.'],
                                            'multi_independent': ['Blood group '
                                                                  'distribution and '
                                                                  'the strength '
                                                                  'breakdown.',
                                                                  'O+ agniveers and '
                                                                  "today's schedule.",
                                                                  'Blood type '
                                                                  'breakdown and the '
                                                                  'pending '
                                                                  'verifications.'],
                                            'simple': ['What is the blood group '
                                                       'distribution?',
                                                       'How many agniveers have blood '
                                                       'group O+?',
                                                       'Show agniveers grouped by '
                                                       'blood type.',
                                                       'Which agniveers are AB '
                                                       'negative?']},
                             'Disease': {'compare': ['Compare disease cases of '
                                                     'Lakhwinder and Thorat company.',
                                                     'Dengue cases in May vs June.',
                                                     'Compare illness cases of platoon '
                                                     '2 and platoon 5.'],
                                         'cross_filter': ['Which agniveers suffering '
                                                          'from dengue are still '
                                                          'marked present today?',
                                                          'Malaria cases among top '
                                                          'BPET performers.',
                                                          'Disease cases among '
                                                          'agniveers currently on '
                                                          'leave.'],
                                         'multi_independent': ['Common diseases and '
                                                               'the current leave '
                                                               'status.',
                                                               'Dengue cases and '
                                                               "today's attendance.",
                                                               'Disease statistics and '
                                                               'the equipment '
                                                               'summary.'],
                                         'simple': ['What are the most common '
                                                    'diseases?',
                                                    'Who is suffering from dengue?',
                                                    'Which agniveers are sick with '
                                                    'fever?',
                                                    'Show me the disease breakdown for '
                                                    'the battalion.']},
                             'Individual': {'compare': ['Compare medical records of '
                                                        'agniveer A0701516F and '
                                                        'agniveer A0701518M.',
                                                        'Health status of agniveer '
                                                        'A0701520K vs agniveer '
                                                        'A0701522P.',
                                                        'Difference between medical '
                                                        'history of agniveer A0701524A '
                                                        'and A0701527N.'],
                                            'cross_filter': ['Medical record of '
                                                             'agniveer A0701583M, who '
                                                             'is also a top BPET '
                                                             'performer.',
                                                             'Health status of '
                                                             'agniveer A0701600F, '
                                                             'currently on leave.',
                                                             'Medical details of '
                                                             'agniveer A0701623H, with '
                                                             'pending verification.'],
                                            'multi_independent': ['Medical record of '
                                                                  'agniveer A0701516F '
                                                                  'and his personal '
                                                                  'details.',
                                                                  'Health status of '
                                                                  'agniveer A0701530N '
                                                                  'and his BPET '
                                                                  'scores.',
                                                                  'Medical history of '
                                                                  'agniveer A0701555A '
                                                                  'and his leave '
                                                                  'record.'],
                                            'simple': ['Show the medical record of '
                                                       'agniveer A0701516F.',
                                                       'What is the health status of '
                                                       'agniveer A0701523X?',
                                                       'Medical history of agniveer '
                                                       'A0701544N.',
                                                       'Give me the medical details of '
                                                       'agniveer A0701567P.']}},
                 'OVERALL': {'OverallPerformance': {'compare': ['Compare overall '
                                                                'performance of '
                                                                'Lakhwinder and '
                                                                'Jaswant company.',
                                                                'Composite ranking of '
                                                                'platoon 2 vs platoon '
                                                                '3.',
                                                                'Compare overall '
                                                                'standing of batch 1 '
                                                                'and batch 2.'],
                                                    'cross_filter': ['Overall '
                                                                     'performance of '
                                                                     'agniveers '
                                                                     'currently on '
                                                                     'leave.',
                                                                     'Composite '
                                                                     'ranking of '
                                                                     'cricket players.',
                                                                     'Overall standing '
                                                                     'of agniveers '
                                                                     'with pending '
                                                                     'verification.'],
                                                    'multi_independent': ['Overall '
                                                                          'performance '
                                                                          "and today's "
                                                                          'schedule.',
                                                                          'Composite '
                                                                          'ranking and '
                                                                          'the leave '
                                                                          'summary.',
                                                                          'Overall '
                                                                          'report and '
                                                                          'the '
                                                                          'equipment '
                                                                          'stats.'],
                                                    'simple': ['Show the overall '
                                                               'performance ranking.',
                                                               'Give me the composite '
                                                               'ranking of all '
                                                               'agniveers.',
                                                               'Who is the best '
                                                               'overall performer?',
                                                               'Show the overall '
                                                               'standing across all '
                                                               'criteria.']}},
                 'PERFORMANCE': {'AttemptWise': {'compare': ['Compare attempt 1 and '
                                                             'attempt 2 scores in '
                                                             'BPET.',
                                                             'Attempt wise scores in '
                                                             'BPET vs PPT.',
                                                             'Compare attempt wise '
                                                             'scores of Lakhwinder and '
                                                             'Jaswant company in '
                                                             'firing.'],
                                                 'cross_filter': ['Attempt wise BPET '
                                                                  'scores for '
                                                                  'agniveers currently '
                                                                  'on leave.',
                                                                  'Scores by attempt '
                                                                  'in PPT for cricket '
                                                                  'players.',
                                                                  'Attempt wise '
                                                                  'breakdown of firing '
                                                                  'for agniveers with '
                                                                  'pending '
                                                                  'verification.'],
                                                 'multi_independent': ['Attempt wise '
                                                                       'scores in BPET '
                                                                       "and today's "
                                                                       'schedule.',
                                                                       'Scores per '
                                                                       'attempt in '
                                                                       'drill and the '
                                                                       'current leave '
                                                                       'status.',
                                                                       'Attempt '
                                                                       'breakdown in '
                                                                       'firing and the '
                                                                       'equipment '
                                                                       'summary.'],
                                                 'simple': ['Show attempt wise scores '
                                                            'for BPET.',
                                                            'Give me the score per '
                                                            'attempt in firing.',
                                                            'Show scores by attempt '
                                                            'for PPT.',
                                                            'Attempt by attempt '
                                                            'breakdown for drill.']},
                                 'Average': {'compare': ['Compare average scores of '
                                                         'Jaswant and Arora company in '
                                                         'BPET.',
                                                         'Average marks in BPET vs '
                                                         'PPT.',
                                                         'Compare average scores of '
                                                         'platoon 1 and platoon 6 in '
                                                         'firing.'],
                                             'cross_filter': ['Average BPET score for '
                                                              'agniveers currently '
                                                              'holding overdue '
                                                              'equipment.',
                                                              'Average marks in PPT '
                                                              'for football players.',
                                                              'Mean firing score for '
                                                              'Dogra class agniveers.'],
                                             'multi_independent': ['Average score in '
                                                                   'BPET and the '
                                                                   'attendance '
                                                                   'summary.',
                                                                   'Average marks in '
                                                                   'drill and who '
                                                                   'plays hockey.',
                                                                   'Mean PPT score and '
                                                                   'the BMI '
                                                                   'distribution.'],
                                             'simple': ['What is the average score in '
                                                        'BPET?',
                                                        'Show me the average marks in '
                                                        'firing.',
                                                        "What's the mean score for "
                                                        'PPT?',
                                                        'On average how did the batch '
                                                        'score in drill?']},
                                 'BestAttempt': {'compare': ['Compare best attempts of '
                                                             'Arora and Thorat company '
                                                             'in BPET.',
                                                             'Personal bests in BPET '
                                                             'vs drill.',
                                                             'Compare best attempt '
                                                             'scores of platoon 3 and '
                                                             'platoon 5 in PPT.'],
                                                 'cross_filter': ['Best attempt in '
                                                                  'BPET for agniveers '
                                                                  'currently on leave.',
                                                                  'Personal bests in '
                                                                  'PPT among kabaddi '
                                                                  'players.',
                                                                  'Best attempt scores '
                                                                  'in firing for fully '
                                                                  'verified '
                                                                  'agniveers.'],
                                                 'multi_independent': ['Best attempt '
                                                                       'in BPET and '
                                                                       'the pending '
                                                                       'verifications.',
                                                                       'Personal bests '
                                                                       'in drill and '
                                                                       'who is present '
                                                                       'today.',
                                                                       'Best attempts '
                                                                       'in firing and '
                                                                       'the latest '
                                                                       'distribution.'],
                                                 'simple': ['What is the best attempt '
                                                            'score of each agniveer in '
                                                            'BPET?',
                                                            "Show me everyone's "
                                                            'personal best in firing.',
                                                            'Highest attempt scores in '
                                                            'PPT.',
                                                            'Best of all attempts in '
                                                            'drill.']},
                                 'Bottom': {'compare': ['Compare bottom performers of '
                                                        'Arora and Thorat company in '
                                                        'BPET.',
                                                        'Lowest scorers in drill vs '
                                                        'firing.',
                                                        'Compare worst performers of '
                                                        'batch 3 and batch 5 in PPT.'],
                                            'cross_filter': ['Bottom performers in '
                                                             'BPET who are also '
                                                             'disqualified.',
                                                             'Lowest scorers in PPT '
                                                             'among agniveers '
                                                             'currently on leave.',
                                                             'Worst performers in '
                                                             'firing among agniveers '
                                                             'with pending '
                                                             'verification.'],
                                            'multi_independent': ['Show bottom '
                                                                  'performers in '
                                                                  'firing and the '
                                                                  'pending '
                                                                  'verifications.',
                                                                  'Lowest scorers in '
                                                                  'BPET and who is '
                                                                  'present today.',
                                                                  'Worst performers in '
                                                                  'PPT and the latest '
                                                                  'distribution.'],
                                            'simple': ['Who are the bottom 10 '
                                                       'performers in BPET?',
                                                       'Show me the lowest scorers in '
                                                       'firing.',
                                                       'Who performed worst in drill?',
                                                       'List the weakest performers in '
                                                       'PPT.']},
                                 'Drop': {'compare': ['Compare score drops of '
                                                      'Lakhwinder and Arora company in '
                                                      'BPET.',
                                                      'Performance drops in drill vs '
                                                      'firing.',
                                                      'Compare declining performers of '
                                                      'batch 2 and batch 6 in PPT.'],
                                          'cross_filter': ['Whose scores dropped in '
                                                           'BPET among agniveers '
                                                           'currently on medical '
                                                           'leave?',
                                                           'Declining performers in '
                                                           'PPT among agniveers with '
                                                           'pending verification.',
                                                           'Score drops in firing for '
                                                           'agniveers who took the '
                                                           'most leave.'],
                                          'multi_independent': ['Whose scores dropped '
                                                                'in BPET and who is '
                                                                'absent today?',
                                                                'Declining performers '
                                                                'in drill and the '
                                                                'equipment summary.',
                                                                'Score drops in PPT '
                                                                'and the list of '
                                                                'absconded agniveers.'],
                                          'simple': ['Whose scores dropped in BPET?',
                                                     'Who declined in performance in '
                                                     'firing?',
                                                     'Which agniveers got worse in PPT '
                                                     'between attempts?',
                                                     'Show me who is sliding down in '
                                                     'drill.']},
                                 'Grading': {'compare': ['Compare grading of Lak and '
                                                         'Jas company in BPET.',
                                                         'Grade distribution in BPET '
                                                         'vs PPT.',
                                                         'Compare excellent grades of '
                                                         'platoon 3 and platoon 4 in '
                                                         'firing.'],
                                             'cross_filter': ['Who got excellent grade '
                                                              'in BPET among agniveers '
                                                              'currently on leave?',
                                                              'Agniveers graded good '
                                                              'in firing among cricket '
                                                              'players.',
                                                              'SAT graded agniveers in '
                                                              'PPT with pending '
                                                              'verification.'],
                                             'multi_independent': ['Who got excellent '
                                                                   'in BPET and '
                                                                   "today's schedule.",
                                                                   'Agniveers graded '
                                                                   'good in drill and '
                                                                   'the current leave '
                                                                   'status.',
                                                                   'Grade wise '
                                                                   'breakdown of '
                                                                   'firing and the '
                                                                   'strength '
                                                                   'breakdown.'],
                                             'simple': ['Who got excellent grade in '
                                                        'BPET?',
                                                        'Show me agniveers graded good '
                                                        'in firing.',
                                                        'Which agniveers got SAT grade '
                                                        'in PPT?',
                                                        'Show the grade wise breakdown '
                                                        'for drill.']},
                                 'GradingSummary': {'compare': ['Compare grading '
                                                                'summary of Lakhwinder '
                                                                'and Thorat company in '
                                                                'BPET.',
                                                                'Grade counts in BPET '
                                                                'vs drill.',
                                                                'Compare grade totals '
                                                                'of batch 1 and batch '
                                                                '4 in PPT.'],
                                                    'cross_filter': ['Grading summary '
                                                                     'of BPET for '
                                                                     'cricket players.',
                                                                     'Grade counts in '
                                                                     'firing for Sikh '
                                                                     'class agniveers.',
                                                                     'Grade tally in '
                                                                     'PPT for '
                                                                     'agniveers '
                                                                     'currently on '
                                                                     'leave.'],
                                                    'multi_independent': ['Grading '
                                                                          'summary of '
                                                                          'BPET and '
                                                                          'who is on '
                                                                          'leave '
                                                                          'today.',
                                                                          'Grade '
                                                                          'counts in '
                                                                          'drill and '
                                                                          'the '
                                                                          'equipment '
                                                                          'stats.',
                                                                          'Grade '
                                                                          'totals in '
                                                                          'firing and '
                                                                          'the pending '
                                                                          'verifications.'],
                                                    'simple': ['Give me the grading '
                                                               'summary for BPET.',
                                                               'How many agniveers are '
                                                               'in each grade in '
                                                               'firing?',
                                                               'Show the grade count '
                                                               'summary for PPT.',
                                                               'What are the grade '
                                                               'totals for drill?']},
                                 'Improvement': {'compare': ['Compare improvement of '
                                                             'Lak and Jas company in '
                                                             'BPET.',
                                                             'Improvement in BPET vs '
                                                             'PPT.',
                                                             'Compare improvement of '
                                                             'platoon 2 and platoon 5 '
                                                             'in firing.'],
                                                 'cross_filter': ['Who improved in '
                                                                  'BPET among '
                                                                  'agniveers currently '
                                                                  'on leave?',
                                                                  'Most improved '
                                                                  'agniveers in PPT '
                                                                  'among cricket '
                                                                  'players.',
                                                                  'Improvement in '
                                                                  'firing for Dogra '
                                                                  'class agniveers.'],
                                                 'multi_independent': ['Who improved '
                                                                       'in BPET and '
                                                                       'who is on '
                                                                       'medical leave?',
                                                                       'Most improved '
                                                                       'in drill and '
                                                                       'the attendance '
                                                                       'summary for '
                                                                       'this month.',
                                                                       'Improvement in '
                                                                       'PPT and the '
                                                                       'blood group '
                                                                       'distribution.'],
                                                 'simple': ['Who improved the most in '
                                                            'BPET?',
                                                            'Which agniveers showed '
                                                            'improvement between '
                                                            'attempts in drill?',
                                                            'Whose scores went up in '
                                                            'PPT?',
                                                            'Show me the most improved '
                                                            'agniveers in firing.']},
                                 'Top': {'compare': ['Compare top performers of '
                                                     'Lakhwinder and Jaswant company '
                                                     'in BPET.',
                                                     'Top scorers in BPET vs PPT.',
                                                     'Compare top performers of '
                                                     'platoon 1 and platoon 2 in '
                                                     'firing.'],
                                         'cross_filter': ['Top performers in BPET who '
                                                          'also play volleyball.',
                                                          'Highest scorers in PPT who '
                                                          'play cricket.',
                                                          'Top 10 in firing among '
                                                          'agniveers currently on '
                                                          'leave.'],
                                         'multi_independent': ['Show top performers in '
                                                               'BPET and who is on '
                                                               'leave today.',
                                                               'Top scorers in PPT and '
                                                               "today's training "
                                                               'schedule.',
                                                               'Who topped firing and '
                                                               'the current equipment '
                                                               'stats.'],
                                         'simple': ['Who are the top 5 performers in '
                                                    'BPET?',
                                                    'Show me the highest scorers in '
                                                    'PPT.',
                                                    'Who topped firing this attempt?',
                                                    'Give me the best performers in '
                                                    'drill.']},
                                 'Trend': {'compare': ['Compare BPET trends of Lak and '
                                                       'Jas company.',
                                                       'Performance trend in BPET vs '
                                                       'PPT.',
                                                       'Compare score trends of batch '
                                                       '2 and batch 5 in firing.'],
                                           'cross_filter': ['BPET performance trend '
                                                            'for agniveers currently '
                                                            'holding overdue '
                                                            'equipment.',
                                                            'Score trend in PPT for '
                                                            'football players.',
                                                            'Performance trend in '
                                                            'firing for agniveers with '
                                                            'pending verification.'],
                                           'multi_independent': ['BPET trend and the '
                                                                 'strength breakdown.',
                                                                 'Performance trend in '
                                                                 'drill and who is on '
                                                                 'leave today.',
                                                                 'Score trend in '
                                                                 'firing and the BMI '
                                                                 'analysis.'],
                                           'simple': ['How has the batch been '
                                                      'performing over attempts in '
                                                      'BPET?',
                                                      'Show the performance trend in '
                                                      'firing.',
                                                      "What's the score pattern over "
                                                      'attempts for PPT?',
                                                      'Show the average marks per '
                                                      'attempt for drill.']}},
                 'PERSONALDETAIL': {'info': {'compare': ['Compare profiles of agniveer '
                                                         'A0701595F and agniveer '
                                                         'A0701596K.',
                                                         'Agniveer A0701602M vs '
                                                         'agniveer A0701603P details.',
                                                         'Difference between the '
                                                         'education of agniveer '
                                                         'A0701611N and A0701612W.'],
                                             'cross_filter': ['Personal details of '
                                                              'agniveers currently on '
                                                              'leave.',
                                                              'Contact info of cricket '
                                                              'players.',
                                                              'Profiles of agniveers '
                                                              'with pending '
                                                              'verification.'],
                                             'multi_independent': ['Personal details '
                                                                   'of agniveer '
                                                                   'A0701561N and his '
                                                                   'medical record.',
                                                                   'Profile of '
                                                                   'agniveer A0701617P '
                                                                   'and his BPET '
                                                                   'scores.',
                                                                   'Biodata of '
                                                                   'agniveer A0701589N '
                                                                   'and his '
                                                                   'verification '
                                                                   'status.'],
                                             'simple': ['Show personal details of '
                                                        'agniveer A0701516F.',
                                                        'Tell me about agniveer '
                                                        'A0701529Y.',
                                                        'Contact information of '
                                                        'agniveer A0701540X.',
                                                        'What is the family background '
                                                        'of agniveer A0701553P?']}},
                 'SCHEDULE': {'Agniveer': {'compare': ['Compare schedules of agniveer '
                                                       'A0701800X and agniveer '
                                                       'A0701801A.',
                                                       'Agniveer A0701805W vs agniveer '
                                                       'A0701807F schedules.',
                                                       'Difference between the '
                                                       'schedules of agniveer '
                                                       'A0701810F and A0701813P.'],
                                           'cross_filter': ['Schedule of agniveer '
                                                            'A0701757F for this week.',
                                                            "Agniveer A0701765A's "
                                                            'schedule for tomorrow.',
                                                            'Personal schedule of '
                                                            'agniveer A0701777P on 15 '
                                                            'July 2026.'],
                                           'multi_independent': ['Schedule of agniveer '
                                                                 'A0701780P and his '
                                                                 'attendance.',
                                                                 "Agniveer A0701788F's "
                                                                 'schedule and his '
                                                                 'leave record.',
                                                                 'Personal schedule of '
                                                                 'agniveer A0701793M '
                                                                 'and his equipment.'],
                                           'simple': ['Schedule for agniveer '
                                                      'A0701725Y.',
                                                      'Show the individual schedule of '
                                                      'agniveer A0701733X.',
                                                      "What is agniveer A0701744K's "
                                                      'personal schedule?',
                                                      'Give me the schedule of '
                                                      'agniveer A0701750X.']},
                              'Company': {'compare': ['Compare schedule of Lakhwinder '
                                                      'and Jaswant company.',
                                                      'Arora versus Thorat company '
                                                      'training schedule.',
                                                      'Compare training agenda of Lak '
                                                      'and Jas company.'],
                                          'cross_filter': ['Lak company schedule for '
                                                           'this week.',
                                                           'Jaswant company training '
                                                           'schedule for tomorrow.',
                                                           'Schedule of Arora company '
                                                           'for batch 3.'],
                                          'multi_independent': ['Schedule for Jas '
                                                                'company and its '
                                                                'strength.',
                                                                "Thorat company's "
                                                                'schedule and its '
                                                                'attendance summary.',
                                                                'Company wise schedule '
                                                                'and the pending '
                                                                'verifications.'],
                                          'simple': ['Schedule for Lakhwinder company.',
                                                     'Show the company wise schedule.',
                                                     "What's the training schedule for "
                                                     'Arora company?',
                                                     "Give me Thorat company's "
                                                     'schedule.']},
                              'Date': {'compare': ['Compare schedules for 2026-07-10 '
                                                   'and 2026-07-15.',
                                                   'Compare the training schedule on '
                                                   '10 July and 12 July.',
                                                   'Schedule on 14 July vs 18 July.'],
                                       'cross_filter': ['Schedule on 10 July 2026 for '
                                                        'Lakhwinder company.',
                                                        'Training plan on 2026-07-12 '
                                                        'for batch 4.',
                                                        'Schedule for 14/07/2026 for '
                                                        'platoon 6.'],
                                       'multi_independent': ['Schedule for 15 July '
                                                             '2026 and the attendance '
                                                             'that week.',
                                                             'Training plan on '
                                                             '2026-07-10 and who is on '
                                                             'leave.',
                                                             'Schedule for 20/07/2026 '
                                                             'and the equipment '
                                                             'stats.'],
                                       'simple': ['Schedule for 15 July 2026.',
                                                  "What's the training schedule on "
                                                  '2026-07-10?',
                                                  'Show the schedule for 20/07/2026.',
                                                  "What's the plan on 12 July 2026?"]},
                              'Today': {'compare': ["Today's vs tomorrow's schedule.",
                                                    "Compare today's schedule of Lak "
                                                    'and Jas company.',
                                                    "Compare today's training plan of "
                                                    'platoon 1 and platoon 2.'],
                                        'cross_filter': ["Today's schedule for "
                                                         'Lakhwinder company.',
                                                         "Today's training plan for "
                                                         'platoon 2.',
                                                         "What's on today for batch "
                                                         '4?'],
                                        'multi_independent': ["Today's schedule and "
                                                              'who is on leave.',
                                                              "Today's training plan "
                                                              'and the attendance for '
                                                              'today.',
                                                              "Today's agenda and the "
                                                              'equipment stats.'],
                                        'simple': ["What's today's schedule?",
                                                   "Show today's training plan.",
                                                   "What's on the agenda today?",
                                                   'Give me the current schedule.']}},
                 'SKILLS': {'ByClass': {'compare': ['Compare Sikh and Dogra class '
                                                    'strength.',
                                                    'Class distribution of batch 1 vs '
                                                    'batch 2.',
                                                    'Compare Jat and Rajput class '
                                                    'rosters.'],
                                        'cross_filter': ['Sikh class agniveers who are '
                                                         'top BPET performers.',
                                                         'Dogra class agniveers '
                                                         'currently on leave.',
                                                         'Class wise roster for '
                                                         'agniveers with pending '
                                                         'verification.'],
                                        'multi_independent': ['Class wise roster and '
                                                              'the strength breakdown.',
                                                              'Sikh class list and the '
                                                              'pending verifications.',
                                                              'Dogra class agniveers '
                                                              'and the equipment '
                                                              'stats.'],
                                        'simple': ['Show agniveers by class.',
                                                   'How many Sikh agniveers are there?',
                                                   'List Dogra class agniveers.',
                                                   'Give me the class wise roster.']},
                            'BySport': {'compare': ['Compare cricket and football '
                                                    'players.',
                                                    'Volleyball vs basketball '
                                                    'participation.',
                                                    'Cricket players in Lak vs Jas '
                                                    'company.'],
                                        'cross_filter': ['Cricket players who are top '
                                                         'BPET performers.',
                                                         'Football players currently '
                                                         'on leave.',
                                                         'Hockey players who are '
                                                         'currently disqualified.'],
                                        'multi_independent': ['Who plays cricket and '
                                                              'who is on leave today?',
                                                              'Football roster and '
                                                              "today's schedule.",
                                                              'Volleyball players and '
                                                              'the BPET toppers.'],
                                        'simple': ['Who plays cricket?',
                                                   'List all football players.',
                                                   'Which agniveers play volleyball?',
                                                   'Show me the sports roster.']}},
                 'STRENGTH': {'StrengthBreakdown': {'compare': ['Compare strength of '
                                                                'Lakhwinder and '
                                                                'Jaswant company.',
                                                                'Headcount of platoon '
                                                                '2 vs platoon 3.',
                                                                'Compare strength of '
                                                                'batch 4 and batch 5.'],
                                                    'cross_filter': ['Strength '
                                                                     'breakdown of '
                                                                     'agniveers '
                                                                     'currently on '
                                                                     'leave.',
                                                                     'Headcount of '
                                                                     'Sikh class '
                                                                     'agniveers.',
                                                                     'Strength of '
                                                                     'agniveers with '
                                                                     'pending '
                                                                     'verification, by '
                                                                     'section.'],
                                                    'multi_independent': ['Strength '
                                                                          'breakdown '
                                                                          "and today's "
                                                                          'schedule.',
                                                                          'Headcount '
                                                                          'and the '
                                                                          'current '
                                                                          'leave '
                                                                          'status.',
                                                                          'Current '
                                                                          'strength '
                                                                          'and the '
                                                                          'equipment '
                                                                          'stats.'],
                                                    'simple': ['Show the strength '
                                                               'breakdown.',
                                                               'What is the current '
                                                               'strength?',
                                                               'Give me the headcount '
                                                               'by section.',
                                                               'How many agniveers are '
                                                               'in each section?']}},
                 'VERIFICATION': {'Completed': {'compare': ['Compare completed '
                                                            'verifications of Lak and '
                                                            'Arora company.',
                                                            'Verified counts in '
                                                            'platoon 2 vs platoon 5.',
                                                            'Compare completed '
                                                            'verifications of batch 3 '
                                                            'and batch 4.'],
                                                'cross_filter': ['Completed '
                                                                 'verifications for '
                                                                 'agniveers who are '
                                                                 'top BPET performers.',
                                                                 'Fully verified '
                                                                 'agniveers currently '
                                                                 'on leave.',
                                                                 'Completed '
                                                                 'verifications for '
                                                                 'Sikh class '
                                                                 'agniveers.'],
                                                'multi_independent': ['Completed '
                                                                      'verifications '
                                                                      'and the BPET '
                                                                      'toppers.',
                                                                      'Verified '
                                                                      'agniveers and '
                                                                      'the attendance '
                                                                      'summary.',
                                                                      'Completed cases '
                                                                      'and the '
                                                                      'disqualified '
                                                                      'list.'],
                                                'simple': ['List completed '
                                                           'verifications.',
                                                           'Which agniveers are fully '
                                                           'verified?',
                                                           'How many verifications are '
                                                           'done?',
                                                           'Show verification cases '
                                                           'that are all clear.']},
                                  'NotResponded': {'compare': ['Compare not responded '
                                                               'cases of Jaswant and '
                                                               'Thorat company.',
                                                               'No response '
                                                               'verifications in '
                                                               'platoon 1 vs platoon '
                                                               '4.',
                                                               'Compare not responded '
                                                               'cases of batch 1 and '
                                                               'batch 6.'],
                                                   'cross_filter': ['Not responded '
                                                                    'verification '
                                                                    'cases among '
                                                                    'agniveers '
                                                                    'currently on '
                                                                    'leave.',
                                                                    'No response '
                                                                    'verifications for '
                                                                    'cricket players.',
                                                                    'Not responded '
                                                                    'cases among '
                                                                    'bottom BPET '
                                                                    'performers.'],
                                                   'multi_independent': ['Not '
                                                                         'responded '
                                                                         'verifications '
                                                                         'and the BMI '
                                                                         'distribution.',
                                                                         'No response '
                                                                         'cases and '
                                                                         'who is '
                                                                         'present '
                                                                         'today.',
                                                                         'Awaiting '
                                                                         'response '
                                                                         'verifications '
                                                                         'and the '
                                                                         'equipment '
                                                                         'stats.'],
                                                   'simple': ['Who has not responded '
                                                              'to verification?',
                                                              'Show verification cases '
                                                              'with no response.',
                                                              "Which agniveers' "
                                                              'verification is '
                                                              'awaiting response?',
                                                              'List unresponsive '
                                                              'verification '
                                                              'requests.']},
                                  'Pending': {'compare': ['Compare pending '
                                                          'verifications of Lakhwinder '
                                                          'and Arora company.',
                                                          'Pending cases in platoon 2 '
                                                          'vs platoon 3.',
                                                          'Compare pending '
                                                          'verifications of batch 1 '
                                                          'and batch 4.'],
                                              'cross_filter': ['Pending verifications '
                                                               'for agniveers '
                                                               'currently holding '
                                                               'overdue equipment.',
                                                               'Pending verification '
                                                               'of cricket players.',
                                                               'Unverified agniveers '
                                                               'among top BPET '
                                                               'performers.'],
                                              'multi_independent': ['Pending '
                                                                    'verifications and '
                                                                    "today's "
                                                                    'attendance.',
                                                                    'Pending cases and '
                                                                    'the strength '
                                                                    'breakdown.',
                                                                    'Awaiting '
                                                                    'verification list '
                                                                    'and the BPET '
                                                                    'toppers.'],
                                              'simple': ['Show pending verifications.',
                                                         'How many verifications are '
                                                         'still pending?',
                                                         'Which agniveers are yet to '
                                                         'be verified?',
                                                         'Who is awaiting '
                                                         'verification?']},
                                  'Rejected': {'compare': ['Compare rejected '
                                                           'verifications of '
                                                           'Lakhwinder and Thorat '
                                                           'company.',
                                                           'Rejected cases in 2025 vs '
                                                           '2026.',
                                                           'Compare denied '
                                                           'verifications of batch 2 '
                                                           'and batch 5.'],
                                               'cross_filter': ['Rejected '
                                                                'verifications among '
                                                                'agniveers who went '
                                                                'AWOL.',
                                                                'Denied verification '
                                                                'cases for agniveers '
                                                                'currently on leave.',
                                                                'Rejected cases among '
                                                                'football players.'],
                                               'multi_independent': ['Rejected '
                                                                     'verifications '
                                                                     'and the '
                                                                     'absconded '
                                                                     'agniveers.',
                                                                     'Denied cases and '
                                                                     "today's "
                                                                     'schedule.',
                                                                     'Rejected '
                                                                     'verification '
                                                                     'list and the '
                                                                     'strength '
                                                                     'breakdown.'],
                                               'simple': ['Show rejected verification '
                                                          'cases.',
                                                          'Whose verification was '
                                                          'denied?',
                                                          'Which agniveers had their '
                                                          'verification turned down?',
                                                          'List all disapproved '
                                                          'verifications.']},
                                  'Sent': {'compare': ['Compare verifications sent in '
                                                       'May and June.',
                                                       'Sent verifications of Lak vs '
                                                       'Jas company.',
                                                       'Compare verification requests '
                                                       'sent for batch 2 and batch 5.'],
                                           'cross_filter': ['Sent verifications for '
                                                            'agniveers currently on '
                                                            'leave.',
                                                            'Verification requests '
                                                            'sent for cricket players.',
                                                            'Sent verifications for '
                                                            'Sikh class agniveers.'],
                                           'multi_independent': ['Sent verifications '
                                                                 'and the strength '
                                                                 'breakdown.',
                                                                 'Verification '
                                                                 'requests sent and '
                                                                 "today's schedule.",
                                                                 'Sent verification '
                                                                 'list and the leave '
                                                                 'summary.'],
                                           'simple': ['How many verification requests '
                                                      'were sent?',
                                                      'Show me the verifications sent '
                                                      'this month.',
                                                      'List requests dispatched for '
                                                      'verification.',
                                                      'Which verification requests '
                                                      'have been sent out?']}}},
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
                       'Jat versus Rajput class rosters.',
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
                            'Bottom performers in PPT from Jat class.',
                            'Average drill score of Rajput class agniveers.',
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
                                 'Personal details of agniveer A0701600F, his medical '
                                 'record, and his BPET scores.',
                                 'Latest distribution, the unassigned agniveers, and '
                                 "today's training plan.",
                                 'Who improved in BPET, the disease statistics, and '
                                 'the rejected verifications.',
                                 'Strength of Lakhwinder company, its attendance '
                                 'summary, and its equipment stats.']}}
