// Load students on page load
document.addEventListener('DOMContentLoaded', function() {
    loadStudents();
});

async function loadStudents() {
    try {
        const response = await fetch('/api/students');
        const students = await response.json();
        
        const select = document.getElementById('studentSelect');
        students.forEach(student => {
            const option = document.createElement('option');
            option.value = student.id;
            option.textContent = student.name;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading students:', error);
    }
}

// Handle student selection
document.getElementById('studentSelect').addEventListener('change', async function() {
    const studentId = this.value;
    if (!studentId) return;
    
    try {
        const response = await fetch(`/api/student/${studentId}`);
        const result = await response.json();
        
        if (!result.success) {
            alert('Error loading student: ' + result.error);
            return;
        }
        
        const data = result.data;
        
        // Fill all form fields
        document.getElementById('age').value = data.age;
        document.getElementById('gender').value = data.gender;
        document.getElementById('ethnicity').value = data.ethnicity;
        document.getElementById('parental_education').value = data.parental_education;
        document.getElementById('parental_support').value = data.parental_support;
        document.getElementById('study_hours').value = data.study_hours.toFixed(1);
        document.getElementById('absences').value = data.absences;
        document.getElementById('tutoring').value = data.tutoring;
        document.getElementById('extracurricular').value = data.extracurricular;
        document.getElementById('sports').value = data.sports;
        document.getElementById('music').value = data.music || 0;
        document.getElementById('volunteering').value = data.volunteering || 0;
        document.getElementById('avg_grade').value = data.avg_grade.toFixed(1);
        document.getElementById('grade_std').value = data.grade_std.toFixed(1);
        document.getElementById('total_courses').value = data.total_courses;
        document.getElementById('failed_courses').value = data.failed_courses;
        document.getElementById('total_credits').value = data.total_credits;
        document.getElementById('stem_ratio').value = data.stem_ratio.toFixed(2);
        document.getElementById('A_count').value = data.A_count;
        document.getElementById('B_count').value = data.B_count;
    } catch (error) {
        console.error('Error loading student data:', error);
        alert('Failed to load student data');
    }
});

// Handle form submission
document.getElementById('predictForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Calculate grade_range and total_activities dynamically
    const avgGrade = parseFloat(document.getElementById('avg_grade').value);
    const gradeStd = parseFloat(document.getElementById('grade_std').value);
    const gradeRange = gradeStd * 2; // Approximate range
    
    const extracurricular = parseInt(document.getElementById('extracurricular').value);
    const sports = parseInt(document.getElementById('sports').value);
    const music = parseInt(document.getElementById('music').value);
    const volunteering = parseInt(document.getElementById('volunteering').value);
    const totalActivities = extracurricular + sports + music + volunteering;
    
    const formData = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        ethnicity: document.getElementById('ethnicity').value,
        parental_education: document.getElementById('parental_education').value,
        parental_support: document.getElementById('parental_support').value,
        study_hours: document.getElementById('study_hours').value,
        absences: document.getElementById('absences').value,
        tutoring: document.getElementById('tutoring').value,
        extracurricular: extracurricular,
        sports: sports,
        total_activities: totalActivities,
        avg_grade: avgGrade,
        total_courses: document.getElementById('total_courses').value,
        failed_courses: document.getElementById('failed_courses').value,
        stem_ratio: document.getElementById('stem_ratio').value,
        total_credits: document.getElementById('total_credits').value,
        grade_std: gradeStd,
        grade_range: gradeRange,
        A_count: document.getElementById('A_count').value,
        B_count: document.getElementById('B_count').value
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (!result.success) {
            alert('Prediction failed: ' + result.error);
            return;
        }
        
        const resultDiv = document.getElementById('result');
        const resultAlert = document.getElementById('resultAlert');
        const resultMessage = document.getElementById('resultMessage');
        const probability = document.getElementById('probability');
        const riskLevel = document.getElementById('riskLevel');
        const recommendation = document.getElementById('recommendation');
        
        resultMessage.textContent = result.message;
        probability.textContent = result.probability_percent;
        riskLevel.textContent = result.risk_level;
        recommendation.textContent = result.recommendation;
        
        resultAlert.className = result.prediction === 1 ? 
            'alert alert-success' : 'alert alert-warning';
        
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction. Please check console for details.');
    }
});
