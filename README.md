# Deep Reinforcement Learning (RL) Using Python

![](https://cdn-images-1.medium.com/max/2560/1*a9F8vOTfpDEM52eW5SSXAQ.jpeg)

## Deep Reinforcement Learning With Python | Part 1 | Creating The Environment

![](https://cdn-images-1.medium.com/max/2000/1*X3zZn4Ic5nFl3a_QyRfbog.gif)

![Left Gif: Explanation of the game rules | | Right Gif: The game played by a human](https://cdn-images-1.medium.com/max/2000/1*y8G1qC-SmTk_cOVDXpnhsQ.gif)

In this tutorial series, we are going through every step of building an expert Reinforcement Learning (RL) agent that is capable of playing games.

This series is divided into three parts:

* **Part 1**: Designing and Building the Game Environment. In this part we will build a game environment and customize it to make the RL agent able to train on it.

* **Part 2**: Build and Train the Deep Q Neural Network (DQN). In this part, we define and build the different layers of DQN and train it.

* **Part 3**: Test and Play the Game.

We might also try making another simple game environment and use Q-Learning to create an agent that can play this simple game.

## The Motivation:

 <iframe src="https://medium.com/media/4f68f7dda1df1fe9363a58257bbab8e2" frameborder=0></iframe>

One time I was in the rabbit hole of YouTube and [THIS VIDEO](https://www.youtube.com/watch?v=k-rWB1jOt9s) was recommended to me, it was about the **sense of self **in human babies, after watching the video a similar question popped into my mind* “Can I develop a smart agent that is smart enough to have a sense of its body and has the ability to change its features to accomplish a certain task?”*

This series is my way of answering this question.

## Designing the Environment:

For this environment, we want the agent to develop a sense of its body and how to change its body features to avoid losing the game.

### First: The Elements of The Environment:

![The Environment](https://cdn-images-1.medium.com/max/2732/1*CWzzRutHg1gYW2V87440JA.jpeg)

**1-** **The Field** contains all the other elements. We represent it in code by class named “Field” as follows:

 <iframe src="https://medium.com/media/891dc4e61fde97cb5cb1c78436f8f81b" frameborder=0></iframe>

**Field** attributes:

* ***width ***(int)**: **the width of the field (not in pixels)

* ***height*** (int: the height of the field (not in pixels)

* ***body ***(np.array)**: **holds the array representation of the game elements (player and walls). This array is passed to the **DQN** and also used to draw the interface using **pygame.**

**Field** methods:

* ***update_field***(self,walls, player) : updates the field.

**2- The Walls :**

 <iframe src="https://medium.com/media/4e045d6da775621cded8b70232960aa3" frameborder=0></iframe>

![The Wall](https://cdn-images-1.medium.com/max/2732/1*5YoIb8kU41MsDecUCIbr3g.jpeg)

**Wall** attributes:

 <iframe src="https://medium.com/media/210bc830ecacdedb452b2a2be8b8bb94" frameborder=0></iframe>

**Wall** methods:

* ***create_hole***(self): Creates a hole in the wall that its width = self.**hole_width.**

* ***move***(self): Moves the wall vertically (every time it get called the wall moves n steps from downward (n = self.speed))

**3- The Player :**

 <iframe src="https://medium.com/media/60be4989fa9fd5cf077833b2bbd4768b" frameborder=0></iframe>

**Player** attributes:

 <iframe src="https://medium.com/media/c6fe9b829d2ecf45db184cbb30469f86" frameborder=0></iframe>

**Player** methods:

* ***move***(self, field, direction = 0 ): Moves the player :
>  - direction = 0 -> No change
>  - direction = 1 -> left
>  - direction = 2 -> right

* ***change_width***(self, action = 0):
>  - action = 0 -> No change
>  - action = 3 -> narrow by one unit
>  - action = 4 -> widen by one unit

## The “**Environment”** Class :

This class facilitates the communication between the environment and the agent, it is designed to with an RL agent or with a human player.

### Main Components Needed by the RL Agent:

1- **ENVIRONMENT_SHAPE attribute: used by the DQN to set the shape of the input layer.**

**2- ACTION_SPACE attribute: used by the DQN to set the shape of the output layer.**

3- **PUNISHMENT** and **REWARD: set the values of both punishment and reward, used to train the agent (we use these values to tell the agent if its previous actions were good or bad).**

4- ***reset*** method: to reset the environment.

5- ***step*** method: takes an action as an argument and returns ***next state*, *reward*, **a boolean variable named **game_over** that is used to tell us if the game is over (the player lost) or not.

It is clear that this environment is not different, it subsumes all the required components and more.

 <iframe src="https://medium.com/media/eb1439d51563c47fa18a84bd1633e75e" frameborder=0></iframe>

**Environment** attributes:

 <iframe src="https://medium.com/media/04aefcd4e61f2eb1a409e3bc8bd98321" frameborder=0></iframe>

**Environment** methods:

* ***__init__***(self) : initializes the environment by initializing some attributes and calling the ***reset ***method.

* ***reset***(self) : resets the environment and returns the state of the game field after resetting it.

* ***print_text***(self, WINDOW = None, text_cords = (0,0), center = False, text = “”, color = (0,0,0), size = 32): prints a text in a given pygame.display (WINDOW) with the given features.

### ***+ step***(self, action):

1- Call the player’s move method to move the player.

2- Call the player’s change_width method to move the player.

3- Move the wall one step.

4- Update the field.

5- Check if the player passed a wall successfully. If so, gives the player a reward and increase its stamina.

6- Check the three losing conditions: the player loses the game if at least one of these three conditions met.

 <iframe src="https://medium.com/media/54a1d80f9ffb5e093dfd94b41db0522f" frameborder=0></iframe>

when a player loses, the value of returned ***reward*** will equal PUNISHMENT, and the indicator of the game state (***game_over***) changes from false to true.

7- Check if the current wall hits the bottom of the field, when that happens, the out of range wall is replaced by a new wall.

8- Return ***next_state*** normalized, ***reward***, ***game_over***

### + render(self, WINDOW = None, human=False):

**Arguments:**

* ***WINDOW*** (pygame.display): the pygame.display that the game will be rendered on.

* ***human*** (bool): If a human will play the game, this argument is set to True, in this case pygame catch pressed keyboard keys to get the action that will be performed.

Explanation of ***render ***method line by line:

1- Check if the player is a human. If so, get the pressed key and translate it to the corresponding action (ex: if the right arrow is pressed then set action = 2, that means move the player on step to the right), then call ***step ***method to perform the chosen action.

2- Update the field then start drawing the walls and the player as blocks.

3- Print the score and the player’s stamina.

4- Finally, update the display to show the rendered screen.

## Finally: Put it all together

Now we are going to use everything we explained and play the game:

The following code repeats the game until the player wins by getting a score higher than or equals ***winning_score, ***or quits the game.

 <iframe src="https://medium.com/media/fb1b83db99342c4b5ad7d396941d773d" frameborder=0></iframe>

You can get the full code [HERE](https://github.com/ModMaamari/reinforcement-learning-using-python).

You can follow me on Twitter [@ModMaamari](https://twitter.com/ModMaamari)

## You may also like:

* [**Deep Neural Networks for Regression Problems](https://medium.com/@mamarih1/deep-neural-networks-for-regression-problems-81321897ca33)**

* [**AI Generates Taylor Swift’s Song Lyrics](https://blog.goodaudience.com/ai-generates-taylor-swifts-song-lyrics-6fd92a03ef7e)**

* [**Introduction to Random Forest Algorithm with Python](https://medium.com/datadriveninvestor/introduction-to-random-forest-algorithm-with-python-9efd1d8f0157)**

* [**Machine Learning Crash Course with TensorFlow APIs Summary](https://medium.com/@mamarih1/machine-learning-crash-course-with-tensorflow-apis-summary-524e0fa0a606)**

* [**How To Make A CNN Using Tensorflow and Keras](https://medium.com/@mamarih1/how-to-make-a-cnn-using-tensorflow-and-keras-dd0aaaed8ab4) ?**

* [**How to Choose the Best Machine Learning Model ?](https://medium.com/@mamarih1/how-to-choose-the-best-machine-learning-model-e1dbb46bdd4d)**
