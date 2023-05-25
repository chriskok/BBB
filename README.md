# Comb

## Setup
- Install conda
- `conda create -n comb python=3.9`
- `pip install -r requirements.txt`
- `python3 -m spacy download en_core_web_lg`
- `python manage.py shell < populate_questions.py`
- `python manage.py shell < populate_answers.py`
- `python manage.py migrate`

## Running the App
- If you just pulled new commits from github: `python manage.py migrate`
- `python manage.py runserver`

## Dev Notes
- When new files are imported: `pip list --format=freeze > requirements.txt`
- Data dumps: `python -Xutf8 manage.py dumpdata --natural-foreign --natural-primary -e contenttypes -e auth.Permission --indent 2 > dump.json`
- Data loading: `python manage.py loaddata dump.json`
- Current superuser: `admin`, pass: `blockpass`
- Accessing GCP Deployment Server: `gcloud compute ssh --zone "us-west2-a" "djangostack-1-vm" --project "comb-grading"`
    - How to leave it running:
        ```
        christopherkok@djangostack-1-vm:~/BBB$ python3 manage.py runserver 0.0.0.0:5000
        CTRL+Z -->    ^Z
        [1]+  Stopped                 python3 manage.py runserver 0.0.0.0:5000
        christopherkok@djangostack-1-vm:~/BBB$ disown -h %1
        christopherkok@djangostack-1-vm:~/BBB$ bg 1
        [1]+ python3 manage.py runserver 0.0.0.0:5000 &
        ```
    - How to remove from bg:
        `ps auxw | grep runserver` -> `kill <pid>`
