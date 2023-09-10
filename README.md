# Анализатор текстовых пресс-релизов

На основании исторических пресс-релизов кредитных рейтинговых агентств участникам хакатона необходимо построить интерпретируемую ML-модель, устанавливающую взаимосвязь между текстом пресс-релиза и присвоенным кредитным рейтингом по национальной рейтинговой шкале Российской Федерации для организации с учетом методологических особенностей оценки рейтинга. ML-модель должна не просто устанавливать соответствие текста пресс-релиза кредитному рейтингу, но также и выделять ключевые конструкции в тексте, соответствующие присвоенному кредитному рейтингу.

## Команда: ЭйАй

**Проблематика:**

- Процесс выставления кредитного рейтинга не автоматизирован
- Человеческий фактор

**Мы предлагаем:**

- Автоматическое выставление рейтинга
- Визуальная интерпретация выставленного рейтинга

**Краткое описание**:

Модель искусственного интеллекта позволяет автоматически получить кредитный рейтинг по полученному тексу пресс релиза.

Модель анализирует текст с помощью Tfidf и прогоняет полученные вектора через XGBoost.

Стэк: Python, Sklearn, transformers, XGBoost

Для локального запуска с сервером необходимо клонировать проект, а также запустить следующие команды (необходимо наличие зависимостей из requirements.txt
):

```
    git clone https://github.com/Yagorka/cb_rf_hackathon
```
Затем перейти в **final_solve.ipynb** с финальным решением и прогнать его

Веса для bert моделей для каждого агенства [здесь](https://drive.google.com/drive/folders/1_icy_b_Ly9OHoU-7RvAehfa2-pnWap-H?usp=sharing)
таблица с результатами [здесь](https://docs.google.com/spreadsheets/d/1J_zI2GweVB0E3VjHf33E073avn-gZXS8Y9yHJGTKKnU/edit?usp=sharing)
