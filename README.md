Welcome to the AMAZING snore detector!!

Have you ever been the only one trying to fall asleep in a room FULL of loud snorers? Not fun. 
In fact, 45% of adults snore at least occasionally! Which means if you haven't been bothered yet, you might be in that situation soon. Yikes.

Never fear! The snore detector is here! This code runs on a Raspberry Pi 4, attached to an OLED screen, GPIO button, speaker, and microphone. 

It's a simple machine learning project. The secret of this pipeline is Mel-Frequency Cepstrum Coefficients, or MFCCs. Think of MFCCs as audio profiles. Every sound has qualities like pitch, volume, timbre, etc. When you extract MFCCs, you assign each of these qualities a number. I took about 10 open source data files of people snoring, extracted MFCCs from them using a Python library called librosa, and put them on a linear regression model. then created a real-time audio pipeline which captures 5 second live audio snippets, compares them to the training data using Euclidian distance (basically the distance formula from middle-school math), and if the distance is below the pre-defined distance threshold, the feedback audio is triggered! Now your snorer-of-choice is awake, no longer snoring, and very annoyed. It's a great way to get back at the people in your life that have snored so loud they have hindered your sleep. 

DISCLAIMER: Use at your own risk. May be extremely cringe. Not recommended on snorers who are easily angered.
