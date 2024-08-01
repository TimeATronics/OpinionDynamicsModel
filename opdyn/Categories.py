"""
    Fuzzy categories for:
    1. Opinion - 5 categories
    2. Average opinion of cells in Confidence Set - 5 categories
    3. NSI Coefficient - 5 categories
"""

fuzzy_cat = {
    "opinion" : {
        "Strongly Disagree": 0.0,
        "Disagree": 0.25,
        "Neutral": 0.5,
        "Agree": 0.75,
        "Strongly Agree": 1.0
    },

    "avg_opinion" : {
        "Strongly Disagree": 0.0,
        "Disagree": 0.25,
        "Neutral": 0.5,
        "Agree": 0.75,
        "Strongly Agree": 1.0
    },

    "nsi_coeff" : {
        "Non-conforming" : 0.0,
        "Slightly non-conforming": 0.25,
        "Neutral": 0.5,
        "Slightly Conforming": 0.75,
        "Conforming": 1.0
    }
}