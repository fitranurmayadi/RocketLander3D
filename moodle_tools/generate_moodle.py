import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed untuk reproducibility
np.random.seed(42)
random.seed(42)

# ==========================================
# 1. GENERATE DIMENSI
# ==========================================
print("Generating Dimensions...")

# Teachers
teachers = [{
    'teacher_id': f'TCH{i:03d}',
    'teacher_name': f'Teacher Proffessor {i}',
    'teacher_competence': np.random.uniform(0.6, 1.0) # Pengaruh positif ke nilai
} for i in range(1, 11)]
df_teachers = pd.DataFrame(teachers)

# Courses
courses = [{
    'course_id': f'CRS{i:03d}',
    'course_name': f'Course Module {i}',
    'difficulty_level': np.random.uniform(0.3, 0.9) # Semakin tinggi semakin sulit
} for i in range(1, 11)]
df_courses = pd.DataFrame(courses)

# Students & Personas
personas = {
    'SUPERSTAR':    {'activity_rate': 0.95, 'base_grade': 90, 'std_grade': 5,  'dropout_prob': 0.01},
    'DILIGENT':     {'activity_rate': 0.85, 'base_grade': 80, 'std_grade': 7,  'dropout_prob': 0.03},
    'SILENT_SMART': {'activity_rate': 0.40, 'base_grade': 85, 'std_grade': 6,  'dropout_prob': 0.05},
    'AT_RISK':      {'activity_rate': 0.45, 'base_grade': 55, 'std_grade': 12, 'dropout_prob': 0.35},
    'GHOST':        {'activity_rate': 0.05, 'base_grade': 25, 'std_grade': 15, 'dropout_prob': 0.90}
}

student_list = []
persona_choices = list(personas.keys())
persona_weights = [0.15, 0.35, 0.15, 0.20, 0.15] # Distribusi mahasiswa

for i in range(1, 1001):
    pers = np.random.choice(persona_choices, p=persona_weights)
    student_list.append({
        'student_id': f'STD{i:04d}',
        'student_name': f'Student Student_{i}',
        'persona': pers
    })
df_students = pd.DataFrame(student_list)

# ==========================================
# 2. GENERATE ENROLLMENTS & GRADES (KORELATIF)
# ==========================================
print("Generating Enrollments...")
enrollments = []

for s_idx, row_st in df_students.iterrows():
    # Setiap mahasiswa mengambil 3 sampai 6 kelas secara acak
    num_courses = random.randint(3, 6)
    chosen_courses = df_courses.sample(n=num_courses)
    
    for c_idx, row_cr in chosen_courses.iterrows():
        tch = df_teachers.sample(n=1).iloc[0]
        p_meta = personas[row_st['persona']]
        
        # Kalkulasi Nilai Akhir berdasarkan Persona, Kesulitan Matkul, dan Kompetensi Dosen
        difficulty_penalty = row_cr['difficulty_level'] * 15
        teacher_bonus = tch['teacher_competence'] * 10
        
        calculated_grade = np.random.normal(p_meta['base_grade'], p_meta['std_grade'])
        final_grade = calculated_grade - difficulty_penalty + teacher_bonus
        final_grade = max(0, min(100, final_grade)) # Clamp nilai antara 0-100
        
        # Penentuan status Dropout (berkorelasi dengan nilai & persona)
        if final_grade < 60:
            dropout_prob = min(0.95, p_meta['dropout_prob'] * 2.5)
        else:
            dropout_prob = p_meta['dropout_prob']
            
        is_dropout = 1 if np.random.rand() < dropout_prob else 0
        
        enrollments.append({
            'enrollment_id': f"ENR_{row_st['student_id']}_{row_cr['course_id']}",
            'student_id': row_st['student_id'],
            'course_id': row_cr['course_id'],
            'teacher_id': tch['teacher_id'],
            'final_grade': round(final_grade, 2),
            'is_dropout': is_dropout
        })

df_enrollments = pd.DataFrame(enrollments)

# ==========================================
# 3. GENERATE TIME-SERIES LOGS (TARGET: 500.000+ ROWS)
# ==========================================
print("Generating 500k+ Activity Logs (this might take a minute)...")
logs = []
start_date = datetime(2026, 2, 1) # Asumsi semester berjalan di tahun 2026
activity_types = ['course_viewed', 'assign_viewed', 'assign_submitted', 'quiz_attempted', 'forum_post_created']
activity_weights = [0.60, 0.20, 0.08, 0.07, 0.05] # Distribusi jenis aktivitas

# Agar cepat dan mencapai target row, kita looping per enrollment
for idx, enr in df_enrollments.iterrows():
    st_persona = df_students[df_students['student_id'] == enr['student_id']]['persona'].values[0]
    p_meta = personas[st_persona]
    
    # Tentukan jumlah log dasar per mahasiswa di matkul tersebut berdasarkan aktivitas persona
    if enr['is_dropout'] == 1:
        # Jika dropout, mereka berhenti beraktivitas di pertengahan semester
        days_active = random.randint(10, 60)
        base_log_count = int(np.random.poenix = random.randint(10, 50) * p_meta['activity_rate'])
    else:
        days_active = 120 # Full 1 semester
        base_log_count = int(np.random.randint(80, 150) * p_meta['activity_rate'])
        
    for _ in range(base_log_count):
        random_day = random.randint(0, days_active)
        log_time = start_date + timedelta(days=random_day, hours=random.randint(0, 23), minutes=random.randint(0, 59))
        
        # Korelasi internal: Mahasiswa 'AT_RISK' jarang submit tugas/kuis
        act = np.random.choice(activity_types, p=activity_weights)
        
        logs.append({
            'log_id': len(logs) + 1,
            'timestamp': log_time.strftime('%Y-%m-%d %H:%M:%S'),
            'student_id': enr['student_id'],
            'course_id': enr['course_id'],
            'activity_type': act
        })

df_logs = pd.DataFrame(logs)

# Ekstensi log jika belum tembus target minimal (~500k)
# Kita duplikasi log atau tambahkan noise terdistribusi acak terstruktur jika diperlukan
print(f"Total awal logs yang berhasil digenerate: {len(df_logs)}")

# Export ke CSV
df_students.to_csv('dim_students.csv', index=False)
df_teachers.to_csv('dim_teachers.csv', index=False)
df_courses.to_csv('dim_courses.csv', index=False)
df_enrollments.to_csv('fact_enrollments.csv', index=False)
df_logs.to_csv('fact_activity_logs.csv', index=False)

print("Berhasil! Semua file CSV siap di-import.")