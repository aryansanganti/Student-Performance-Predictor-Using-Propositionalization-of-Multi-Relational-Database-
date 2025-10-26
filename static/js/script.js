// Load students on page load
document.addEventListener('DOMContentLoaded', function () {
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
document.getElementById('studentSelect').addEventListener('change', async function () {
    const studentId = this.value;
    if (!studentId) return;

    try {
        const response = await fetch(`/api/student/${studentId}`);
        const data = await response.json();

        document.getElementById('age').value = data.age;
        document.getElementById('gender').value = data.gender;
        document.getElementById('study_hours').value = data.study_hours.toFixed(1);
        document.getElementById('avg_grade').value = data.avg_grade.toFixed(1);
        document.getElementById('total_courses').value = data.total_courses;
        document.getElementById('failed_courses').value = data.failed_courses;
        document.getElementById('stem_ratio').value = data.stem_ratio.toFixed(2);
        document.getElementById('total_credits').value = data.total_credits;
        document.getElementById('grade_std').value = data.grade_std.toFixed(1);
    } catch (error) {
        console.error('Error loading student data:', error);
    }
});

// Handle form submission
document.getElementById('predictForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = {
        age: document.getElementById('age').value,
        gender: document.getElementById('gender').value,
        study_hours: document.getElementById('study_hours').value,
        avg_grade: document.getElementById('avg_grade').value,
        total_courses: document.getElementById('total_courses').value,
        failed_courses: document.getElementById('failed_courses').value,
        stem_ratio: document.getElementById('stem_ratio').value,
        total_credits: document.getElementById('total_credits').value,
        grade_std: document.getElementById('grade_std').value
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

        const resultDiv = document.getElementById('result');
        const resultAlert = document.getElementById('resultAlert');
        const resultMessage = document.getElementById('resultMessage');
        const probability = document.getElementById('probability');

        resultMessage.textContent = result.message;
        probability.textContent = (result.probability * 100).toFixed(1) + '%';

        resultAlert.className = result.prediction === 1 ?
            'alert alert-success' : 'alert alert-warning';

        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth' });
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction. Please try again.');
    }
});
