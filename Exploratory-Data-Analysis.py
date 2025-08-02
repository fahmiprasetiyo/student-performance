import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_csv('student-mat.csv', sep=';')
except Exception as e:
    print(f"Error loading CSV: {e}")
    try:
        df = pd.read_csv('student-mat.csv')
        print("Semicolon separator failed, but comma separator worked.")
    except Exception as e2:
        print(f"Fallback to comma separator also failed: {e2}")
        # Exit if the file can't be loaded
        exit()

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Data Inspection
print("--- Data Info ---")
df.info()
print("\n--- Descriptive Statistics ---")
print(df.describe())
print("\n--- First 5 Rows ---")
print(df.head())



# 1. Distribution of Final Grades (G3)
plt.figure(figsize=(12, 6))
sns.histplot(df['G3'], bins=20, kde=True, color='blue')
plt.title('Distribusi Nilai Akhir (G3)', fontsize=16)
plt.xlabel('Nilai Akhir (G3)', fontsize=12)
plt.ylabel('Jumlah Siswa', fontsize=12)
plt.savefig('G3_distribution.png')
plt.close()

# 2. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sex', data=df, palette='viridis')
plt.title('Distribusi Siswa Berdasarkan Jenis Kelamin', fontsize=16)
plt.xlabel('Jenis Kelamin', fontsize=12)
plt.ylabel('Jumlah Siswa', fontsize=12)
plt.savefig('gender_distribution.png')
plt.close()

# 3. Mother's Job Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Mjob', data=df, palette='plasma', order = df['Mjob'].value_counts().index)
plt.title('Distribusi Pekerjaan Ibu', fontsize=16)
plt.xlabel('Pekerjaan Ibu', fontsize=12)
plt.ylabel('Jumlah Siswa', fontsize=12)
plt.savefig('mjob_distribution.png')
plt.close()

# 4. Father's Job Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Fjob', data=df, palette='plasma', order = df['Fjob'].value_counts().index)
plt.title('Distribusi Pekerjaan Ayah', fontsize=16)
plt.xlabel('Pekerjaan Ayah', fontsize=12)
plt.ylabel('Jumlah Siswa', fontsize=12)
plt.savefig('fjob_distribution.png')
plt.close()

# 5. Study Time vs. Final Grade
plt.figure(figsize=(12, 7))
sns.boxplot(x='studytime', y='G3', data=df, palette='cyan')
plt.title('Waktu Belajar Mingguan vs. Nilai Akhir', fontsize=16)
plt.xlabel('Waktu Belajar (1: <2 jam, 2: 2-5 jam, 3: 5-10 jam, 4: >10 jam)', fontsize=12)
plt.ylabel('Nilai Akhir (G3)', fontsize=12)
plt.savefig('studytime_vs_g3.png')
plt.close()

# 6. Past Failures vs. Final Grade
plt.figure(figsize=(12, 7))
sns.boxplot(x='failures', y='G3', data=df, palette='magenta')
plt.title('Jumlah Kegagalan Kelas Sebelumnya vs. Nilai Akhir', fontsize=16)
plt.xlabel('Jumlah Kegagalan', fontsize=12)
plt.ylabel('Nilai Akhir (G3)', fontsize=12)
plt.savefig('failures_vs_g3.png')
plt.close()

# 7. Internet Access vs. Final Grade
plt.figure(figsize=(8, 6))
sns.barplot(x='internet', y='G3', data=df, palette='autumn')
plt.title('Akses Internet di Rumah vs. Nilai Akhir Rata-rata', fontsize=16)
plt.xlabel('Punya Akses Internet?', fontsize=12)
plt.ylabel('Rata-rata Nilai Akhir (G3)', fontsize=12)
plt.savefig('internet_vs_g3.png')
plt.close()

# 8. Correlation Heatmap
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(18, 15))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 10})
plt.title('Peta Korelasi Antar Variabel Numerik', fontsize=18)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()


print("\nEDA visualizations have been generated and saved as PNG files.")
