if __name__ == '__main__':
    file = open("./generated_phrases.txt", 'r')

    lines = file.readlines()

    for line in lines:
        string = ''
        previous_char = None
        for char in line:
            if char == '[' or char == ']' or char == '\'' or char == ',':
                pass
            elif char == ' ':
                if previous_char != ',':
                    string += ' '
            else:
                string += char

            previous_char = char

        print(string)

    file.close()
