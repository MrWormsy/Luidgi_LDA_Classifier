# Luidgi_LDA_Classifier

In order to run it you must install all the libraries with :
```{shell}
pip install -r requirements.txt
```

And then if you want to use the Luidi's monitoring website you must type
```{shell}
luigid &
python MainLuigi.py --scheduler-host localhost StepMain
```

Else you can simply type if you want to run it alone
```{shell}
python MainLuigi.py --local-scheduler StepMain
```
