Write-Host "================== Testing GET / (Root) ================="
$root = Invoke-RestMethod -Uri http://127.0.0.1:5000/ -Method GET
Write-Host ($root | ConvertTo-Json -Depth 5)

Write-Host "`n================== Testing GET /health =================="
$health = Invoke-RestMethod -Uri http://127.0.0.1:5000/health -Method GET
Write-Host ($health | ConvertTo-Json -Depth 5)

Write-Host "`n================== Testing POST /predict ================"
$body = @{
    gender = "Male"
    country = "United States"
    occupation = "Corporate"
    self_employed = "No"
    family_history = "No"
    days_indoors = "1-14 days"
    growing_stress = "Yes"
    changes_habits = "Yes"
    mental_health_history = "No"
    mood_swings = "Medium"
    coping_struggles = "No"
    work_interest = "Yes"
    social_weakness = "No"
    mental_health_interview = "No"
    care_options = "Not sure"
} | ConvertTo-Json

$resp = Invoke-RestMethod -Uri http://127.0.0.1:5000/predict -Method POST -ContentType "application/json" -Body $body
Write-Host ($resp | ConvertTo-Json -Depth 5)
