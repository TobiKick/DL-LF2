"LittleFighter2" Project for the course Deep Learning Applications at National Taipei University of Technology


------------------------------------------------------------------------------------------------------------------

Write code in the app.py file

Let it run with "python app.py"

------------------------------------------------------------------------------------------------------------------

To start
Env
Make an LF2 environment.
Note: The web driver will be closed automatically when the process exits.

import lf2gym
env = lf2gym.make()
All parameters for make() are described here.

Server
Or if you simply want to run a LF2Server.

import lf2gym
lf2gym.start_server(port=8000)
Open your browser, and connect to http://127.0.0.1:8000/game/game.html to play LF2!



Keyboard Control
Action	Player 1	Player 2
Up	W	Up (U)
Right	D	Right (K)
Down	X	Down (M)
Left	A	Left (H)
Attack	S	J (J)
Jump	Q	U (I)
Defense	Z	M (,)
