import pytest

feedback_greter_100_words = ("The first time I used it, the app seemed good to me. Afterwards," 
"I started it to use it daily and started realizing certain annoying errors such as the music"
"being outdated, there are no local bands in my area and also that it cannot be used without internet."
"On the other hand, I value certain functionalities such as being able to share my list of favorite"
"songs on different social networks just by clicking and how easy and intuitive it was to learn to use it."
"I really do not think I will pay for the premium subscription for the things mentioned but I am sure I will" 
"continue to be a regular user of the free version as long as it stays that way. I hope some day this features"
"will be solved and the user could have a most pleasure exoerience, until that I will mantain my posture"
"mentioned earlier and still use my cds for my favourite local bands")

@pytest.mark.parametrize("feedback, rating, prediction, status_code", [
    (None, 1, "prediction", 422), 
    ("good app", 4, "prediction", 400),
    (feedback_greter_100_words, 3, "prediction", 400),
    ("this is a very good and fun app", 6, "prediction", 400),
    ("Very good app. I would use it again, i really enjoy it", 5, "Positive", 201),
    ("Very bad app. I would not use it again, i really dislike it", 1, "Negative", 201),
    ("Muy buena la aplicacion, la voy a seguir usando todos los dias", 1, "prediction", 400),
])
def test_create_feedback(client, feedback, rating, prediction, status_code):
    res = client.post(
        "/feedback/", json={"feedback":feedback , "rating": rating, "prediction":prediction})

    assert res.status_code == status_code

