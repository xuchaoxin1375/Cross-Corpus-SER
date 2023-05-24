[toc]

## release note

- for convenient,I tag the version of the project (mainly on the client ) with data time
- the time(version) was create by powershell :

  - ```powershell
    Get-Date -Format "yyyy-MM-dd@HH:mm:ss"
    ```
  
  - copy to clipbord automatically
  
    ```bash
    Get-Date -Format "yyyy-MM-dd@HH:mm:ss"|scb
    ```
  
    
  
  

### versions with time

- **<u>input version under this line:</u>**
  - new notes:"Sort update records in descending order of time".That would be convenient to update new version notes in the future.
- *<u>current modifing:</u>*
  - ...
  - improve the UI alignment(make almost all `input` and `frame` type elements using the expand_x=True,to autofit the width availible)
  - fix the audio file viewer language inconsistent bugs.
  
- 2023-05-10@20:06:02
  - initially support the interface language switch:between Chinese and English
    - the approch to achieve the multilanguage support depending on language string resource py file:
      - `zh.py` for Chinese
      - `en.py` for english
    - for convenience,these module was managed by the `multilanguage.py`module.
    - unfortunately,these features was test in `fviewer.py`,the main gui was still unable to work normally because the `windows.close` caused the program broken!
  - improve the regex_filter performance by using `re.compile` instead `re.<function>`
- 2023-05-09@19:10:05
  - fix some bugs.
  - certain adjustments to the project code.

### old notes

- old notes:"Sort update records in ascending order of time".

- 2023-04-25@21:35:50
  - new:
    - update the layout of the client
    - use the table element to show "test_score","train_score" and "predict_proba" if it is usable
    - update the file viwer part function and operation logic with more flexibility.
  - kwon problems:
    - with the display or not display the predict_proba layout,the left column may need to resize the window to make the last part of view to visible!

- 2023-04-27@22:04:52
  - "update the client layout with more available parameters cross-validation(`kfold`/`shufflesplit`/`StratifiedShuffleSplit`)
  - support the `n_splits` as a slider element for use to adjust th
  - update the `confusion matrix table`show button 

- 2023-04-28@11:27:17
  - "update the make_window function in the ccser_client main file with more clear modular extraction"
- 2023-04-30@00:02:03
  - "update PCA transformer to the SER system"
  - "there are some problems with recognize the single audio with pca preprocessing,the idea is to save the pca transformer when it was first fited,and then try to load corresponding file(joblib or pickle) when you need to recognize several audios that need extract the required dimension feature"
- 

