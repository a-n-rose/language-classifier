## Test Ur Speech

To test your speech, install the dependencies located in the local installations.md file as well as those in the main directory. 

The program expects a sound to indicate to users that recording is starting. This wavefile should be located in a 'soundfiles' subdirectory and the path saved in the script 'id_ur_speech_func.py' in the function 'play_go()'.

Save the models you would like to classify your speech with in a subdirectory named 'models'.

Check global variables - e.g. where your speech data shoud be saved.

Run 'id_ur_speech.py' 
```
(env)...$ python3 id_ur_speech.py
```

### ToDo
* Change where get_date() gets called/how users' recordings get saved - right now the user could record their speech over and over but their recordings would get recored over. 
* Check 'find speech start' functions - for some reason that's not working as well as it used to... at least when I listen to saved wavefiles. But it still identified that no speech was present... 
