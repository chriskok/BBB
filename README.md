# Block-by-Block (BBB)

## Setup
- Install conda
- `conda create -n bbb python=3.9`
- `pip install -r requirements.txt`
- `python manage.py shell < populate_questions.py`
- `python manage.py shell < populate_answers.py`
- `python manage.py migrate`

## Running the App
- If you just pulled new commits from github: `python manage.py migrate`
- `python manage.py runserver`

## Dev Notes
- When new files are imported: `pip list --format=freeze > requirements.txt`
- Data dumps: `python -Xutf8 manage.py dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.Permission --indent 2 > dump.json`
- Current superuser: `admin`, pass: `blockpass`