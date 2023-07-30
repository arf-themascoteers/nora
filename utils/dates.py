from datetime import datetime

datetime_str = '01-Apr-2022'

datetime_object = datetime.strptime(datetime_str, '%d-%b-%Y')

print(datetime_object.timestamp())
print(int(datetime_object.timestamp()))