# Defining the Player, Jersey, and Equipment classes as per the assignment requirements.

class Player:
    def __init__(self, team, name, height, weight, age, equipment):
        self.team = team
        self.name = name
        self.height = height
        self.weight = weight
        self.age = age
        self.equipment = equipment

    def __str__(self):
        return (f"Player(Name: {self.name}, Team: {self.team}, Height: {self.height}, "
                f"Weight: {self.weight}, Age: {self.age}, Equipment: {self.equipment})")


class Jersey:
    def __init__(self, number, colors, size, logo, name_on_back):
        self.number = number
        self.colors = colors
        self.size = size
        self.logo = logo
        self.name_on_back = name_on_back

    def __str__(self):
        return (f"Jersey(Number: {self.number}, Colors: {self.colors}, Size: {self.size}, "
                f"Logo: {self.logo}, Name on Back: {self.name_on_back})")



class Equipment:
    def __init__(self, helmet, jersey, pads, cleats):
        self.helmet = helmet
        self.jersey = jersey
        self.pads = pads
        self.cleats = cleats

    def __str__(self):
        return (f"Equipment(Helmet: {self.helmet}, Jersey: {self.jersey}, "
                f"Pads: {self.pads}, Cleats: {self.cleats})")


# List to hold all the Player objects, simulating a PlayerAssociation.
players_list = []


# Function to add a player to the players list.
def add_player(player):
    players_list.append(player)
    return len(players_list) - 1  # Returning the index of the newly added player.


# Function to edit a player's details.
def edit_player(index, new_player = None):
    op = int(input('select the index of the player you want to edit: '))
    player = players_list[op]
    op1 = input('select the attributes you want to edit: ')
    print('1. Team')
    print('2. Name')
    print('5. Retire')
    print('6. sub')
    if op1 == '1':
        player.team = input('Enter the new team name: ')

    elif op1 == '2':
        player.name = input('Enter the new name: ')

    elif op1 == '3':
        player.equipment.jersey.number = input('Enter the new number: ')

    elif op1 == '5':
        players_list.pop(op)
        print('Player has been retired')
        return
    elif op1 == '6':
        players_list.pop(op)
        print('Player has been subbed')
        players_list.append(new_player)
        return

# Function to print all player details.
def print_players():
    for player in players_list:
        print(player)


# Function to find a player by jersey number.
def find_by_jersey_number(number):
    list = []
    for player in players_list:
        if player.equipment.jersey.number == number:
            list.append(player)
    return list

# Function to find players by their team name.
def find_by_team(team_name):
    list = []
    for player in players_list:
        if player.team == team_name:
            list.append(player)
    return list


# Example usage of the functions and classes.
# Creating a Jersey and Equipment object to use in our Player object.
test_jersey = Jersey(number=99, colors=['Red', 'Black'], size='L', logo='Bull', name_on_back='Jordan')
test_equipment = Equipment(helmet='Red Helmet', jersey=test_jersey, pads='Large Pads', cleats='Speedy Cleats')

# Creating a Player object.
test_player = Player(team='Bulls', name='Michael', height=198, weight=98, age=30, equipment=test_equipment)

# Adding the player to our players list.
player_index = add_player(test_player)

# Printing out all players.
print_players()

# Searching for players with a specific jersey number.
players_with_number_99 = find_by_jersey_number(99)

# Searching for players in a specific team.
players_in_bulls_team = find_by_team('Bulls')

edit_player(0)