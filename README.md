# Machine Learning for Finance

Public repository associated with the Machine Learning for Finance course.

## Labs

The `labs` directory contains class lab code and data.

## Projects

### Objective

The objective of this group project is to apply machine learning techniques to the financial industry (any asset class, any problem - look beyond price prediction if needed). Regardless of the chosen track, strong emphasis should be placed on:

- **Problem:**
    - What problem is the group trying to solve? (e.g., price prediction, portfolio allocation, stock selection, order management)
    - How can it be translated into a machine learning problem? State the problem clearly.

- **Data:**
    - How is the dataset built?
    - What does it look like?
    - Does it need cleaning or pre-processing?
    - What features are interesting to build?
    - How can it be visualized?

- **Model:**
    - What naive models can be used on this problem as a benchmark?
    - What models can better solve the problem and why?

- **Questions:**
    - What issues are raised by the data and modeling approach?
    - How can we iterate on data and model to try and solve them?

### Different Tracks

#### Alternative Data

In this track, I will be interested in projects that use data that is not natively quantitative to enrich the problem or model used.

Here are a few examples:
- **Text:** news, tweets, online reviews, public filings
- **Image:** satellite images for earnings predictions
- **LLM-generated analysis** on financial reports
- **Sound analysis** on a Fed's speech

#### Complex Models

In this track, I will be interested in projects that use advanced models and analyze in-depth their parameters, training, tuning, etc., to achieve the best or most robust results. Emphasis should be placed on the complexity/accuracy trade-off, and naive benchmarks should be considered.

Examples:
- Deep learning models (transformers, LSTM, NeuralProphet)
- Reinforcement Learning

#### Free-Ride

If you are very interested in a topic regarding Machine Learning and Finance and it does not fit in the above categories, feel free to **suggest** it to me.

Examples:
- Implement a technical article of your choice whose implementation is not available online
- Participate in a trading or machine-learning challenge

### Constraints

- **Group size:** 3 members (edge cases: 4 or 2×2)
- **Track choice:** Choose one of the 3 above tracks
- **Code documentation:** Code must be documented (docstrings, explicit variable names, etc.) following PEP 8
- **Written presentation:** Must be independent from code (I don't want to reach out to students for code errors). Examples: HTML output of Jupyter notebook, or a PPT explicitly linked to code
- **Oral presentation:** 7 minutes EXACTLY (going over will be penalized). Should focus on the most important aspect of your project (not all of it), worth showing other students and discussing in class

### Evaluation

#### General

The project should demonstrate a scientific process. How you think carefully about the data you have and how to model them in regards to the objective you describe will be the main reason for success or failure.

#### Code

- Is the main code (except potentially Jupyter for presentation) in plain Python (*.py extension)?
- Is it readable and understandable?
    - Explicit variable names
    - Comments
    - Docstrings
    - Pythonic style
- Is it concise and clear? (i.e., no LLM-generated garbage)

#### Written Presentation

- Is the problem clearly stated?
- How is the group approaching the problem?
- How is data processed, cleaned, and visualized?
- What are the project's underlying assumptions?
- What are the project's main findings?
- What issues did the group face and how did they solve them?
- What could be next steps, or acknowledged blind spots?

#### Oral Presentation

- Finishes on time (7 minutes)
- States clearly one interesting artifact or finding from the project
- Answers questions from the class and teacher

### Timeline

- **Groups:** Form groups by February 9th
- **Code submission:** Code committed and packaged into a single PR → main by March 30th
- **PR to main:** Use the structure as in `projects/EXAMPLE`, with a `README.md` with some info, and a clear project structure. Use as many subdirectories as you need. Replace `EXAMPLE` with the underscore-joined names of your group members (e.g., `lagadec_lagadec_lagadec` if I partnered with my brother and sister)

### Problems

- **Never used Git?** Reach out and/or partner with seasoned coders
- **Questions?** Reach out via email: dorian.lagadec@gmail.com - include `[ml_for_finance]` in the subject line

### How to Find Data

Data can be hard to find for free. Here are a few resources:

- **Yahoo Finance**
- **Kaggle** — past competitions usually keep their data open
- **APIs:** X (Twitter), Meta, News API
- **Student resources:** As a student, you might be allowed free data sources through PSL
- **Challenge Data ENS**
