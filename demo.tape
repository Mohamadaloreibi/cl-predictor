# Spelar in en GIF av terminalen till README:n.
# Installera:  brew install vhs      Kör:  vhs demo.tape
Output assets/demo.gif

Set FontSize 14
Set Width 1400
Set Height 900
Set Theme "Dracula"
Set TypingSpeed 60ms

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 1s
Show

Type "python cl_predict.py --demo"
Sleep 500ms
Enter
Sleep 32s