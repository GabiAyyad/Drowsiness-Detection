import pygame

# Initialize pygame mixer
pygame.mixer.init()
pygame.mixer.music.load(r"C:\Users\User\Desktop\Projects\Drowsiness-Detection\assets\Alarm.mp3")  # your .wav/mp3 file
pygame.mixer.music.play()

print("Alarm playing... press Enter to stop")
input()  # wait so the sound can play
pygame.mixer.music.stop()
