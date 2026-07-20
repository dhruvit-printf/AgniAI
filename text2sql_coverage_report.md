# Text2SQL Coverage Report

This report reflects the current code-path behavior in the AgniAI SQL pipeline.
I attempted an offline sweep with the real executor, but the full run timed out before completing, so this file is based on the current executor/intent rules plus the fallback buckets that were already confirmed during the partial run.

## Falls Back To Text2SQL

### Attendance
- `Show this week's attendance.`
- `Give me the weekly attendance report.`
- `Show this month's attendance.`
- `Give me the monthly attendance summary.`
- `Show the attendance summary.`
- `Give me the overall attendance report.`

### Medical
- `Show the blood group of Agniveer A0701882L.`
- `Show the blood group report.`
- `Show all diagnosed Agniveers.`
- `Show the disease report.`

### Equipment
- `Show the equipment of Agniveer A0701882L.`
- `List the issued items of Agniveer A0701882L.`

### Distribution
- `Show the latest distribution.`
- `Display the latest unit allocation.`
- `Which unit has the highest number of Agniveers?`
- `Show the unit with the maximum strength.`

## Stays Deterministic

### Performance
All listed Performance questions now stay on the deterministic executor path.

### Attendance
- `Show today's attendance.`
- `Who is absent today?`
- `Who is present today?`
- `Show all Agniveers currently on campus.`
- `Show the current strength.`
- `What is the total manpower strength?`

### Leave
All listed Leave questions stay deterministic.

### Medical
- `Show the BMI report.`
- `What is the BMI of Agniveer A0701882L?`
- `Show the medical report of Agniveer A0701882L.`
- `Display the complete health record of Agniveer A0701882L.`

### Equipment
- `Show the equipment summary.`
- `Give me the equipment statistics.`
- `Show equipment by category.`
- `List all rifle equipment.`
- `Show all returned equipment.`
- `Which equipment has been returned?`
- `Show the currently issued equipment.`
- `Which equipment is currently with Agniveers?`

### Verification
All listed Verification questions stay deterministic.

### Distribution
- `Show the distribution by unit.`
- `Give me the unit-wise distribution report.`
- `Show all unassigned Agniveers.`
- `Which Agniveers have not been assigned to any unit?`

### Personal Details
All listed Personal Details questions stay deterministic.

## Notes

- BMI is now computed from `MedicalRecordMaster` plus fallback `AgniveerMaster.Height` / `AgniveerMaster.Weight`, not a stored `BmiValue`.
- The text2sql fallback prompt now includes the project overview, schema map, and executor map so it mirrors the codebase instead of inventing generic SQL.
- If you want, I can turn this into a CSV or JSON matrix next so you can diff it against future code changes.
