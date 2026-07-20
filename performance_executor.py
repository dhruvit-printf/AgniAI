
import logging
from typing import Dict, Any, List, Optional, Tuple
from sql_executor import run_readonly, _to_section, SQL_MAX_ROWS

logger = logging.getLogger("performance_executor")

def execute_performance_query(intent: Dict) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    category = intent.get("category", "")
    operation = intent.get("operation", "")
    
    # Common filters
    batch_id = intent.get("batchId")
    agniveer_class = intent.get("class")
    platoon_id = intent.get("platoonId")
    company_id = intent.get("companyId")
    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    section = str(intent.get("section") or "").strip()
    sub_section = str(intent.get("sub_section") or "").strip()
    attempt_no = intent.get("attemptNo") or intent.get("attempt_no")
    
    args = []
    
    # Reusable Scoped-Agniveer EXISTS filter
    scoped_exists = '''
    EXISTS (
        SELECT 1 FROM AgniveerMaster m
        WHERE m.Id = {outer_alias}.AgniveerId
          AND m.IsActive = 1
          AND ISNULL(m.IsDisqualified,0) = 0
          {batch_filter}
          {class_filter}
          {platoon_filter}
          {company_filter}
          {agniveer_filter}
    )'''
    
    batch_filter = ""
    if batch_id is not None:
        batch_filter = "AND m.BatchId = ?"
        args.append(int(batch_id))
        
    class_filter = ""
    if agniveer_class is not None:
        class_filter = "AND LOWER(m.Class) = LOWER(?)"
        args.append(str(agniveer_class))
        
    platoon_filter = ""
    if platoon_id is not None:
        platoon_filter = "AND m.PlatoonId = ?"
        args.append(int(platoon_id))
        
    company_filter = ""
    if company_id is not None:
        company_filter = "AND EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = m.PlatoonId AND p.CompanyId = ?)"
        args.append(int(company_id))
        
    agniveer_filter = ""
    if agniveer_no is not None:
        agniveer_filter = "AND LOWER(m.AgniveerNo) LIKE '%' + LOWER(?) + '%'"
        args.append(str(agniveer_no))

    def get_scoped_exists(outer_alias: str, is_nested=False) -> str:
        res = scoped_exists.format(
            outer_alias=outer_alias,
            batch_filter=batch_filter,
            class_filter=class_filter,
            platoon_filter=platoon_filter,
            company_filter=company_filter,
            agniveer_filter=agniveer_filter
        )
        return res

    def append_scoped_args(target_args: list):
        if batch_id is not None: target_args.append(int(batch_id))
        if agniveer_class is not None: target_args.append(str(agniveer_class))
        if platoon_id is not None: target_args.append(int(platoon_id))
        if company_id is not None: target_args.append(int(company_id))
        if agniveer_no is not None: target_args.append(str(agniveer_no))

    if operation in ("Top", "Bottom", "OverallPerformance", "BestAttempt"):
        # Q1/Q2 logic
        top_n = intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS
        descending = False if operation == "Bottom" else True
        order_dir = "DESC" if descending else "ASC"
        
        q_args = []
        if attempt_no is not None:
            attempt_filter = "sa.AttemptNo = ?"
            q_args.append(str(attempt_no))
        else:
            attempt_filter = "sa.IsBestAttempt = 1"
            
        if sub_section:
            sql = f'''
            SELECT a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal 
            FROM AgniveerScoreAttempt sa
            JOIN AgniveerMaster a ON a.Id = sa.AgniveerId
            WHERE sa.SubItemId IN (SELECT si.Id FROM ScoreSubItemMaster si WHERE LOWER(si.Name) LIKE '%' + LOWER(?) + '%')
              AND sa.MarksObtained IS NOT NULL
              AND {attempt_filter}
              AND {get_scoped_exists('sa')}
            GROUP BY a.AgniveerNo, a.FullName
            ORDER BY BestTotal {order_dir}
            '''
            q_args.insert(0, sub_section)
        else:
            if section:
                sec_filter = "(LOWER(sec.SectionName) LIKE '%' + LOWER(?) + '%' AND ISNULL(sec.IsExceptional,0)=0)"
                q_args.insert(0, section)
            else:
                sec_filter = "ISNULL(sec.IsExceptional,0)=0"
                
            sql = f'''
            SELECT a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal 
            FROM AgniveerScoreAttempt sa
            JOIN AgniveerMaster a ON a.Id = sa.AgniveerId
            WHERE sa.MarksObtained IS NOT NULL
              AND {attempt_filter}
              AND EXISTS (
                  SELECT 1 FROM ScoreSubItemMaster si JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
                  WHERE si.Id = sa.SubItemId AND {sec_filter}
              )
              AND {get_scoped_exists('sa')}
            GROUP BY a.AgniveerNo, a.FullName
            ORDER BY BestTotal {order_dir}
            '''
            
        append_scoped_args(q_args)
        
        try:
            rows, err = run_readonly(sql, tuple(q_args))
            if err: return None, err
            top_n = int(top_n)
            rows = rows[:top_n] if rows else []
            section_payload = _to_section(rows=rows, intent=intent, sql=sql)
            return section_payload, None
        except Exception as e:
            return None, str(e)
            
    elif operation in ("Grading", "GradingSummary"):
        try:
            top_n = intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS
            requested_grade = intent.get("grading")
            
            sec_sql = "SELECT Id, SectionName FROM ScoreSectionMaster WHERE ISNULL(IsExceptional,0) = 0"
            sec_args = []
            if section:
                sec_sql += " AND LOWER(SectionName) LIKE '%' + LOWER(?) + '%'"
                sec_args.append(section)
                
            sec_rows, err = run_readonly(sec_sql, tuple(sec_args))
            if err: 
                print(f"[DEBUG SEC ERR] {err}", flush=True)
                return None, err
            if not sec_rows: return _to_section(rows=[], intent=intent, sql=sec_sql), None
            
            target_section_ids = [r["Id"] for r in sec_rows]
            section_map = {r["Id"]: r["SectionName"] for r in sec_rows}
            
            q_marks = ",".join(["?"] * len(target_section_ids))
            si_sql = f"SELECT Id, SectionId, MaxMarks FROM ScoreSubItemMaster WHERE SectionId IN ({q_marks})"
            si_rows, err = run_readonly(si_sql, tuple(target_section_ids))
            if err: 
                print(f"[DEBUG SI ERR] {err}", flush=True)
                return None, err
            subitem_map = {r["Id"]: {"SectionId": r["SectionId"], "MaxMarks": r["MaxMarks"]} for r in si_rows}
            target_subitem_ids = list(subitem_map.keys())
            if not target_subitem_ids: return _to_section(rows=[], intent=intent, sql=si_sql), None
            
            ag_sql = f'''
            SELECT Id, AgniveerNo, FullName
            FROM AgniveerMaster m
            WHERE m.IsActive = 1 AND ISNULL(m.IsDisqualified,0) = 0
              {batch_filter} {class_filter} {platoon_filter} {company_filter} {agniveer_filter}
            '''
            ag_args = []
            append_scoped_args(ag_args)
            ag_rows, err = run_readonly(ag_sql, tuple(ag_args))
            if err: 
                print(f"[DEBUG AG ERR] {err}", flush=True)
                return None, err
            ag_map = {r["Id"]: r for r in ag_rows}
            scoped_ag_ids = list(ag_map.keys())
            if not scoped_ag_ids: return _to_section(rows=[], intent=intent, sql=ag_sql), None
            
            q_ags = ",".join(["?"] * len(scoped_ag_ids))
            q_sis = ",".join(["?"] * len(target_subitem_ids))
            sa_sql = f'''
            SELECT sa.AgniveerId, sa.SubItemId, sa.MarksObtained
            FROM AgniveerScoreAttempt sa
            WHERE sa.IsBestAttempt = 1 AND sa.MarksObtained IS NOT NULL
              AND sa.AgniveerId IN ({q_ags}) AND sa.SubItemId IN ({q_sis})
            '''
            sa_args = scoped_ag_ids + target_subitem_ids
            sa_rows, err = run_readonly(sa_sql, tuple(sa_args))
            if err: 
                print(f"[DEBUG SA ERR] {err}", flush=True)
                return None, err
            
            import collections
            ag_sec_group = collections.defaultdict(lambda: collections.defaultdict(list))
            for row in sa_rows:
                ag_id, si_id = row["AgniveerId"], row["SubItemId"]
                sec_id = subitem_map[si_id]["SectionId"]
                ag_sec_group[ag_id][sec_id].append(row)
                
            def get_grade(pct: float) -> str:
                if pct >= 90: return 'Exceptionally Well'
                if pct >= 75: return 'Excellent'
                if pct >= 60: return 'Good'
                if pct >= 45: return 'SAT'
                return 'Fail'
                
            results = []
            if operation == "GradingSummary":
                grade_counts = collections.defaultdict(lambda: collections.defaultdict(int))
                for ag_id, sec_dict in ag_sec_group.items():
                    for sec_id, rows in sec_dict.items():
                        obtained = sum(r["MarksObtained"] for r in rows)
                        distinct_si = set(r["SubItemId"] for r in rows)
                        dyn_max = sum(subitem_map[si]["MaxMarks"] for si in distinct_si)
                        pct = round((obtained / dyn_max * 100), 2) if dyn_max > 0 else 0
                        grade_counts[sec_id][get_grade(pct)] += 1
                
                for sec_id, counts in grade_counts.items():
                    sec_name = section_map[sec_id]
                    for grade in ['Exceptionally Well', 'Excellent', 'Good', 'SAT', 'Fail']:
                        if counts[grade] > 0:
                            results.append({"SectionName": sec_name, "Grade": grade, "Count": counts[grade]})
                results.sort(key=lambda x: x["SectionName"])
                return _to_section(rows=results, intent=intent, sql=sa_sql), None
            else:
                for ag_id, sec_dict in ag_sec_group.items():
                    ag_info = ag_map[ag_id]
                    if section:
                        for sec_id, rows in sec_dict.items():
                            obtained = sum(r["MarksObtained"] for r in rows)
                            distinct_si = set(r["SubItemId"] for r in rows)
                            dyn_max = sum(subitem_map[si]["MaxMarks"] for si in distinct_si)
                            pct = round((obtained / dyn_max * 100), 2) if dyn_max > 0 else 0
                            grade = get_grade(pct)
                            if requested_grade and grade.lower() != requested_grade.lower(): continue
                            results.append({"AgniveerNo": ag_info["AgniveerNo"], "FullName": ag_info["FullName"], "SectionName": section_map[sec_id], "Percentage": pct, "Grade": grade})
                    else:
                        total_obtained = 0
                        overall_distinct_si = set()
                        for sec_id, rows in sec_dict.items():
                            total_obtained += sum(r["MarksObtained"] for r in rows)
                            overall_distinct_si.update(r["SubItemId"] for r in rows)
                        dyn_max = sum(subitem_map[si]["MaxMarks"] for si in overall_distinct_si)
                        pct = round((total_obtained / dyn_max * 100), 2) if dyn_max > 0 else 0
                        grade = get_grade(pct)
                        if requested_grade and grade.lower() != requested_grade.lower(): continue
                        results.append({"AgniveerNo": ag_info["AgniveerNo"], "FullName": ag_info["FullName"], "OverallPercentage": pct, "Grade": grade})
                
                top_n = int(top_n)
                sort_key = "Percentage" if section else "OverallPercentage"
                results.sort(key=lambda x: x[sort_key], reverse=True)
                return _to_section(rows=results[:top_n], intent=intent, sql=sa_sql), None
        except Exception as e:
            return None, str(e)

    return None, f"Operation {operation} logic not yet fully translated in executor."
