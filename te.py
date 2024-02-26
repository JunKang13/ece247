filename = "ActivityLog.txt"
file = open(filename, "w+")
activities = file.read()


class instrument:
    def __init__(self, instrument_name, play_time, music_genre):
        self.play_time = play_time
        self.music_genre = music_genre
        self.instrument_name = instrument_name

    def __str__(self):
        return f"I have been playing a music instrument called {self.instrument_name} for {self.play_time} years, I like to play {self.music_genre} with this instrument."


class pet:
    def __init__(self, pet_type, pet_time, pet_number):
        self.pet_time = pet_time
        self.pet_type = pet_type
        self.pet_number = pet_number

    def __str__(self):
        return f"I have had {self.pet_number} {self.pet_type}(s) in my home for {self.pet_time} years."


class video_game:
    def __init__(self, game_name, game_type, game_cost, game_platform):
        self.game_name = game_name
        self.game_cost = game_cost
        self.game_type = game_type
        self.game_platform = game_platform

    def __str__(self):
        return f"I just bought a video game called {self.game_name} on {self.game_platform}. It's a(n) {self.game_type} and only cost me {self.game_cost} dollars."


MyInstrument = instrument("Erhu", 12, "Chinese traditional music")
MyPet = pet("dog", 4, 2)
MyGame = video_game("Hollow Knight", "indie game", 10, "Steam")

file = open(filename, "w")
file.write(str(MyInstrument) + "\n")
file.write(str(MyPet) + "\n")
file.write(str(MyGame) + "\n")
file.close()

file = open(filename, "r")
print(file.read())
file.close()