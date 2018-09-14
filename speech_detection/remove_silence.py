from prep_speech import Voice_Start_Finish


wavename = 'somewave.wav'

speech = Voice_Start_Finish(wavename)

speech.plot_original_signal()
speech.plot_speechonly_signal()
speech.plot_energy()
speech.plot_speechonly_energy()
speech.plot_power()
speech.plot_speechonly_power()

print("Plotting of original and speech only data successfully completed and saved.")
