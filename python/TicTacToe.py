import os, random

def print_board(board, players=None, instructions=False):
	#This function prints out the board that it was printed
	clear_screen = os.system('cls' if os.name == 'nt' else 'clear')

	#if instructions are to be displayed
	if (instructions == True):
		print('Welcome to Tic Tac Toe! \n')
	else:	
		print('Computer - ' + players[1] + '\tPlayer' + players[0] + '\n')

	# "board" is a list of 10 characters representing the board(ignoring index 0)
	for i in xrange (1,9,3):
		print('     |  |')
		print('   ' + board[i] + ' |' + board[i+1] + ' |' + board[i+2])
		print('     |  |')
		print('   ------------')

def input_player_letter():
	#assigns x or 0 for player letter. Return list with [player_letter, computer_letter]
	print('Do you want to be X or 0?')
	letter = raw_input().upper()
	while not (letter == 'X' or letter == '0'):
		print('Invalid input. Do you want to be X or 0?')
		letter = raw_input().upper()

	if letter =='X':
		return ['X', '0']
	else:
		return ['0', 'X']

def toss():
	#Toss to find who goes first, user or computer
	if random.randint(0, 1) == 0:
		return 'computer'
	else:
		return 'player'

def play_again():
	#returns whether user wants to continue playing.
	print('Do you want to play Again? (Y/N)')
	return not raw_input().lower().startswith('n')

def input_player_move(board):
	#Let the player type in his move.
	print('What is your next move? (1-9)')
	move= raw_input()
	while move not in '1 2 3 4 5 6 7 8 9'.split() or not isEmpty(board, int(move)):
		print('Invalid input ! Please mark your move again.(1-9)')
		move= raw_input()
	return int(move)

def is_board_full(board):
	#return true if every space on board(except index 0) is non-empty.
	for i in range(1,10):
		if isEmpty(board, i):
			return False
	return True

def isEmpty(board, move):
	# Return True if the passed move index in board list is empty.
	return board[move]==' '

def check_win(board,letter):
	#Returns true if three continous letters found on board.
	return ((board[7] == letter and board[8] == letter and board[9] == letter) or #bottom row
	(board[4] == letter and board[5] == letter and board[6] == letter) or			#middle row	
	(board[1] == letter and board[2] == letter and board[3] == letter) or			#top row
	(board[1] == letter and board[4] == letter and board[7] == letter) or
	(board[2] == letter and board[5] == letter and board[8] == letter) or
	(board[3] == letter and board[6] == letter and board[9] == letter) or
	(board[3] == letter and board[5] == letter and board[7] == letter) or
	(board[1] == letter and board[5] == letter and board[9] == letter)) #right diagonal

def choose_random_move_from_list(board, moves_list):
	#Returns a valid move from the passed list on the passed board.
	#Returns none if there is no valid move.
	possible_moves=[]
	for i in moves_list:
		if isEmpty(board, i):
			possible_moves.append(i)

	if len(possible_moves)!=0:
		return random.choice(possible_moves)
	else:
		return None

def get_computer_move(board, players):
	#Given a board and the computer's letter, determine where to move and return that move.
	player_letter, computer_letter = players

	#Tic Tac Toe AI:
	#Win : Check if computer can win on next move
	for i in range(1,10):
		copy_board=board[:]
		if isEmpty(copy_board, i):
			copy_board[i] = computer_letter
			if check_win(copy_board, computer_letter):
				return i
	#Block : check if player could win on next time
	for i in range(1,10):
		copy_board=board[:]
		if isEmpty(copy_board, i):
			copy_board[i] = computer_letter
			if check_win(copy_board, player_letter):
				return i			
	#Random moves
	#choose a random free corner
	move = choose_random_move_from_list(board, [1,3,7,9])
	if move!=None:
		return move

	#choose center if free
	if isEmpty(board, 5):
		return 5

	#choose random free edge centre	
	move = choose_random_move_from_list(board, [2,4,6,8])

if __name__ =="__main__":
	while True:
		board=[' '] + '1 2 3 4 5 6 7 8 9'.split()
		print_board(board, None, True)
		board = [' '] * 10
		players = input_player_letter()
		player_letter, computer_letter = players
		turn=toss()
		print('The'+ turn +'will go first.\nPress enter to continue.')
		raw_input()
		game_over= False

		while not game_over:
			if turn == 'player':
				print_board(board, players)
				move=input_player_move(board)
				board[move] = player_letter

				if check_win(board, player_letter):
					print_board(board, players)
					print('Congratulation! YOu have won the game!')
					game_over=True
				else:
					if is_board_full(board):
						print_board(board,players)
						print('The game is a tie!')
						break
					else:
						turn = 'computer'

			else:
				move = get_computer_move(board,players)
				board[move]	= computer_letter

				if check_win(board, computer_letter):
					print_board(board, players)
					print('The computer won the game.')
					game_over=True
				else:
					if is_board_full(board):
						print_board(board,players)
						print('The game is a tie!')
						break
					else:
						turn = 'player'
		if not play_again():
			break				