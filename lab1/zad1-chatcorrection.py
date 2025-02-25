import datetime
import math

def get_user_data():
    name = input("Your name: ")
    year = int(input("Year of birth: "))
    month = int(input("Month of birth: "))
    day = int(input("Day of birth: "))
    return name, datetime.datetime(year, month, day)

def calculate_days_lived(birth_date):
    current_date = datetime.datetime.now()
    return abs((current_date - birth_date).days)

def calculate_biorhythm(days, cycle):
    return math.sin(2 * math.pi / cycle * days)

def predict_next_day_biorhythm(current_days, cycle):
    return calculate_biorhythm(current_days + 1, cycle)

def check_biorhythm(name, value, days, cycle):
    if value > 0.5:
        print(f"Your {name} coefficient is: {value:.2f}. Congrats, you are in good shape!")
    elif value < -0.5:
        print(f"Your {name} coefficient is: {value:.2f}. Ahh, don't be sad!")
        next_day_value = predict_next_day_biorhythm(days, cycle)
        if next_day_value > value:
            print(f"Tomorrow your {name} parameter will be better!")
        else:
            print(f"Tomorrow your {name} parameter will be worse :c")

def main():
    name, birth_date = get_user_data()
    days_lived = calculate_days_lived(birth_date)
    print(f"Hi {name}, today is the {days_lived}th day of your life!")
    
    biorhythms = {
        "physical": (23, calculate_biorhythm(days_lived, 23)),
        "emotional": (28, calculate_biorhythm(days_lived, 28)),
        "intellectual": (33, calculate_biorhythm(days_lived, 33))
    }
    
    for bio_name, (cycle, value) in biorhythms.items():
        check_biorhythm(bio_name, value, days_lived, cycle)

if __name__ == "__main__":
    main()

#kod napisany przez chat GPT jak widac jest bardziej czytelny, funkcje sa krotsze i bardziej zrozumiale
