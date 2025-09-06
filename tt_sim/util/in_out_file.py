def read_binary_from_file(filename):
    with open(filename, "rb") as file:
        data = file.read()
        print(type(data))
