# Video-Engagement
To demonstrate machine learning and coding skills


Skills demonstrated: classification, scikit-learn, TensorFlow, model evaluation, general coding

This code trains one or more classifiers to assess how engaging videos from a dataset are based on seven features defined below. A video is classified as "engaging" if the median percentage of the video watched across all viewers was at least 30%.

Dataset: engagement_data.csv

Features:

1. title_word_count - the number of words in the title of the video.

2. document_entropy - a score indicating how varied the topics are covered in the video, based on the transcript. Videos with smaller entropy scores will tend to be more cohesive and more focused on a single topic.

3. freshness - The number of days elapsed between 01/01/1970 and the lecture published date. Videos that are more recent will have higher freshness values.

4. easiness - A text difficulty measure applied to the transcript. A lower score indicates more complex language used by the presenter.

5. fraction_stopword_presence - A stopword is a very common word like 'the' or 'and'. This feature computes the fraction of all words that are stopwords in the video lecture transcript.

6. speaker_speed - The average speaking rate in words per minute of the presenter in the video.

7. silent_period_rate - The fraction of time in the lecture video that is silence (no speaking).

Target variable:

1. engagement - Target label for training. True if learners watched a substantial portion of the video (see description), or False otherwise.
'''
