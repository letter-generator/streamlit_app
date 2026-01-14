import matplotlib
import matplotlib.pyplot as plt

# Данные
sources = ['OpenAlex', 'ArXiv', 'Semantic Scholar']
percentages = [40.1, 32.5, 27.4]  # 100 - (40.1 + 32.5) = 27.4

# Цвета для секторов
colors = ['#ff9999', '#66b3ff', '#99ff99']

# Создание диаграммы
fig, ax = plt.subplots(figsize=(8, 6))
wedges, texts, autotexts = ax.pie(
    percentages,
    labels=sources,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=(0.00, 0.00, 0),  # Немного отделяем первые два сектора
    shadow=True,
    textprops={'fontsize': 12}
)

# Делаем проценты белыми и жирными
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Добавляем заголовок
plt.title('Распределение по источникам', fontsize=16, fontweight='bold', pad=20)

# Делаем диаграмму круглой
ax.axis('equal')

# Добавляем легенду (опционально)
ax.legend(wedges, 
          [f'{label}: {pct}%' for label, pct in zip(sources, percentages)],
          title="Источники",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

# Показываем диаграмму
plt.tight_layout()
plt.show()

# Для сохранения в файл раскомментируйте следующую строку:
# plt.savefig('pie_chart.png', dpi=300, bbox_inches='tight')