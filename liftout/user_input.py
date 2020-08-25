"""User input functions."""


def ask_user(message, default=None):
    """Ask the user a question and return True if they say yes.

    Parameters
    ----------
    message : str
        The question to ask the user.
    default : str, optional
        If the user presses Enter without typing an answer,
        the default indicates how to interpret this.
        Choices are 'yes' or 'no'. The efault is None.

    Returns
    -------
    bool
        Returns True if the user answers yes, and false if they answer no.
    """
    yes = ["yes", "y"]
    no = ["no", "n"]
    if default:
        if default.lower() == "yes":
            yes.append("")
        elif default.lower() == "no":
            no.append("")
    all_posiible_responses = yes + no
    user_response = "initial non-empty string"
    while user_response not in all_posiible_responses:
        user_response = input(message)
        if user_response.lower() in yes:
            return True
        elif user_response.lower() in no:
            return False
        else:
            print("Please enter 'yes' or 'no'")
