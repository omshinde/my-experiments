import random

def playAgain():
	print('Do you want to play again? (Y/N)')
	return not input().lower().startswith('n')

if __name__=="__main__":

	while True:

		diceNumber = random.randint(1,6)
		print('Enter a number between 1 and 6. \nIf your entry matches with the lucky number, then you win it!!')
		userEntry=int(input())
		print('You entered: %s'%userEntry)
		if (diceNumber == userEntry):
			print('Your entry and the Lucky number is same i.e: %s '%diceNumber)
			print('You win!!')
		else:
			print('Your entry is %s and the lucky number is %s'%(userEntry, diceNumber))
			print('You lose!')

		if not playAgain():
			break	 
