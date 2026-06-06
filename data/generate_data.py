"""
Synthetic Email Dataset Generator
Generates 1000+ realistic spam and ham emails for training the spam classifier.
"""

import csv
import os
import random

# ─── Spam Templates ──────────────────────────────────────────────────────────

SPAM_SUBJECTS = [
    "You've WON", "URGENT", "Congratulations", "Limited Time Offer",
    "Act Now", "FREE", "Exclusive Deal", "Don't Miss Out",
    "Claim Your Prize", "Special Promotion"
]

SPAM_BODIES = [
    # Lottery / Prize scams
    "Congratulations! You have been selected as the winner of our international lottery. "
    "Click the link below to claim your prize of $1,000,000. "
    "This offer expires in 24 hours so act fast!",

    "URGENT: Your email has been randomly selected for a cash prize of $500,000. "
    "To claim your winnings, reply with your full name, address, and bank details immediately. "
    "Do not share this email with anyone.",

    "You are the lucky winner of our weekly sweepstakes! "
    "A sum of $250,000 has been allocated to your name. "
    "Click here to verify your identity and receive your funds within 48 hours.",

    "Dear Winner, we are pleased to inform you that your email won $750,000 in our annual draw. "
    "Send us your personal details to process the payment. "
    "This is a one-time opportunity, don't let it slip away!",

    "CONGRATULATIONS! You've been chosen to receive a brand new iPhone 15 Pro Max absolutely FREE! "
    "Just complete a short survey and provide your shipping address. "
    "Hurry, only 10 units left!",

    # Free offers / Deals
    "Get a FREE laptop today! No purchase necessary. "
    "Simply click the link and enter your email to qualify. "
    "This exclusive offer is available for a limited time only.",

    "AMAZING DEAL: Buy one get three FREE on all designer watches! "
    "Use code FREESHIP for free worldwide shipping. "
    "Shop now before stocks run out!",

    "FREE trial of our premium weight loss supplement! "
    "Lose 30 pounds in 30 days guaranteed or your money back. "
    "Order now and pay only shipping and handling.",

    "Exclusive offer just for you! Get 90% off on all luxury handbags. "
    "This flash sale ends tonight at midnight. "
    "Click here to browse our collection and save big!",

    "You've been pre-approved for a $50,000 credit line! "
    "No credit check required. Apply now by clicking the link below. "
    "Limited spots available, first come first served!",

    # Phishing attempts
    "ALERT: Your bank account has been compromised. "
    "Click here immediately to verify your identity and secure your account. "
    "Failure to act within 24 hours will result in account suspension.",

    "Your PayPal account has been limited due to suspicious activity. "
    "Please log in through the secure link below to restore full access. "
    "Ignoring this notice will result in permanent account closure.",

    "IMPORTANT: Your Netflix subscription payment failed. "
    "Update your billing information now to avoid service interruption. "
    "Click here to update your payment method securely.",

    "Dear customer, your Amazon order #38291 has been placed successfully. "
    "If you did not make this purchase of $899.99, click here to cancel immediately. "
    "Your account security is our top priority.",

    "WARNING: Unusual sign-in activity detected on your Microsoft account. "
    "Someone tried to access your account from Russia. "
    "Click here to change your password and enable two-factor authentication.",

    # Money / Investment scams
    "Make $5,000 per day working from home! No experience needed. "
    "Our proven system guarantees results in just 7 days. "
    "Join thousands of successful members today!",

    "BITCOIN ALERT: Invest $250 today and earn $10,000 by next week! "
    "Our AI-powered trading bot does all the work for you. "
    "Don't miss this once-in-a-lifetime opportunity!",

    "Secret investment strategy revealed! Wall Street insiders don't want you to know this. "
    "Turn $100 into $50,000 in just 30 days. "
    "Click here to learn the method that's making ordinary people rich!",

    "Earn passive income of $3,000 weekly with our automated forex system. "
    "No trading experience required. Start with just $50. "
    "Join over 100,000 successful traders worldwide!",

    "URGENT BUSINESS PROPOSAL: I am Dr. James from Nigeria. "
    "I have $15.5 million that I need to transfer out of the country. "
    "I will give you 30% of the total if you help me. Reply for details.",

    # Medication / Health spam
    "Buy discount medications online! Up to 80% off on all prescriptions. "
    "No prescription needed. Fast discreet shipping worldwide. "
    "Order now and get free bonus pills!",

    "BREAKTHROUGH: Scientists discover miracle cure for baldness! "
    "Regrow a full head of hair in just 2 weeks. "
    "Order your FREE trial bottle today, just pay shipping!",

    "Lose weight fast with our revolutionary diet pill! "
    "Celebrities are raving about this amazing product. "
    "Get your 30-day supply FREE. Limited stock available!",

    "Amazing anti-aging cream turns back the clock 20 years! "
    "Dermatologists hate this one simple trick. "
    "Order now and get 50% off plus free shipping!",

    "Get bigger muscles in just 7 days with our new supplement! "
    "Used by professional athletes worldwide. "
    "Buy 2 bottles get 1 FREE. No side effects guaranteed!",

    # Urgency / Fear tactics
    "FINAL WARNING: Your computer has been infected with 47 viruses! "
    "Download our security software immediately to protect your data. "
    "Click here for a FREE emergency scan right now!",

    "Your email account will be DELETED in 24 hours unless you verify. "
    "Click the verification link below to confirm your identity. "
    "This is an automated message from the IT security team.",

    "CRITICAL: Your social security number has been compromised! "
    "Call our toll-free number immediately to freeze your credit. "
    "Failure to act now could result in identity theft!",

    "Last chance! Your exclusive membership discount expires in 1 hour. "
    "Save 95% on premium software licenses today only. "
    "This deal will NOT be repeated. Act now or regret forever!",

    "ATTENTION: You have an unclaimed tax refund of $3,247.00! "
    "File your claim online within 48 hours or the refund will be forfeited. "
    "Click here to submit your banking details for direct deposit.",
]

# ─── Ham Templates ────────────────────────────────────────────────────────────

HAM_BODIES = [
    # Work / Professional emails
    "Hi team, just a reminder that we have our weekly standup meeting tomorrow at 10 AM. "
    "Please prepare your status updates and any blockers you'd like to discuss. "
    "See you all in the conference room.",

    "Hey, I've pushed the latest changes to the feature branch. "
    "Could you please review the pull request when you get a chance? "
    "The main updates are in the authentication module.",

    "Good morning everyone. The quarterly report is ready for review. "
    "I've attached the PDF to this email. "
    "Please share your feedback by Friday end of day.",

    "Hi, I wanted to follow up on our discussion about the new project timeline. "
    "I've updated the Gantt chart with the revised milestones. "
    "Let me know if the dates work for your team.",

    "Reminder: The office will be closed next Monday for the holiday. "
    "Please make sure all urgent tasks are completed by Friday. "
    "Enjoy the long weekend!",

    "Hey, I noticed a bug in the payment processing module. "
    "When users enter an invalid card number, the error message doesn't display correctly. "
    "I've created a ticket for it. Can you take a look?",

    "Hi all, we're organizing a team building event next Thursday. "
    "It'll be an escape room challenge followed by dinner. "
    "Please RSVP by Wednesday so I can finalize the reservation.",

    "Thanks for your help with the client presentation yesterday. "
    "The client was really impressed with the demo. "
    "They've scheduled a follow-up meeting for next Tuesday.",

    "Hi, I've completed the code review for the API endpoints. "
    "Overall it looks great, just a few minor suggestions in the comments. "
    "Nice work on the error handling!",

    "Good afternoon. The server migration is scheduled for this Saturday at 2 AM. "
    "Expected downtime is approximately 4 hours. "
    "I'll send out notifications to all affected users.",

    # Personal / Family emails
    "Hey! How are you doing? It's been a while since we caught up. "
    "Want to grab coffee this weekend? "
    "Let me know what works for you.",

    "Hi Mom, I'm doing well. Work has been busy but good. "
    "I'm planning to visit next month for the holidays. "
    "Can't wait to see everyone!",

    "Happy birthday! Hope you have an amazing day filled with joy and happiness. "
    "Wishing you all the best for the year ahead. "
    "Let's celebrate this weekend!",

    "Hey, thanks for the recipe you sent last week. I tried making it and it turned out great! "
    "The kids absolutely loved it. "
    "Do you have any other easy dinner ideas?",

    "Hi, just wanted to let you know that the family reunion is confirmed for August 15th. "
    "We've booked the lakehouse for the weekend. "
    "Please let everyone know and bring your favorite dish!",

    "Hey, I saw the photos from your vacation. Looks like you had an amazing time! "
    "Where exactly did you stay? I'd love to plan a similar trip. "
    "We should catch up soon and you can tell me all about it.",

    "Hi, I'm dropping off the kids at school tomorrow morning. "
    "Can you pick them up at 3 PM? "
    "I have a dentist appointment that I can't reschedule.",

    "Hey, just finished reading that book you recommended. It was fantastic! "
    "The plot twist at the end completely caught me off guard. "
    "Any other recommendations?",

    # Academic / Educational
    "Dear Professor, I wanted to confirm my attendance for the research seminar next week. "
    "I've prepared a brief summary of my findings so far. "
    "Would it be possible to schedule a 15-minute slot for my presentation?",

    "Hi class, the assignment deadline has been extended to next Wednesday. "
    "Please make sure to include proper citations and references. "
    "Office hours are available on Monday if you need help.",

    "Hey, do you have the notes from yesterday's lecture? "
    "I had to leave early for a doctor's appointment. "
    "I'd really appreciate it if you could share them.",

    "Dear students, the midterm exam will cover chapters 1 through 7. "
    "I've uploaded a practice test on the course portal. "
    "Review sessions will be held next Tuesday and Thursday.",

    # Shopping / Services (legitimate)
    "Thank you for your order! Your package has been shipped and is expected to arrive by Thursday. "
    "You can track your delivery using the link in your account. "
    "If you have any questions, please contact our support team.",

    "Hi, your appointment with Dr. Smith is confirmed for next Monday at 2:30 PM. "
    "Please arrive 15 minutes early to complete the paperwork. "
    "If you need to reschedule, call us at least 24 hours in advance.",

    "Your monthly subscription has been renewed successfully. "
    "The charge of $9.99 has been applied to your credit card. "
    "You can manage your subscription settings in your account.",

    "Dear member, our gym will be undergoing renovations starting next month. "
    "The pool area will be closed for two weeks but all other facilities remain open. "
    "We apologize for any inconvenience.",

    # News / Updates
    "Here's your weekly digest of top stories. "
    "The tech industry saw major developments with new AI regulations. "
    "Read the full articles on our website.",

    "Hi, I wanted to share this interesting article about sustainable energy. "
    "It discusses how solar panel efficiency has improved by 40% in the last decade. "
    "Thought you might find it relevant for your research.",

    "Good news! The community garden project has been approved by the city council. "
    "We'll be starting the planting season next month. "
    "Volunteers can sign up at the community center.",

    "Hey, did you see the game last night? What an incredible comeback! "
    "That last-minute goal was unbelievable. "
    "We should watch the next match together at the sports bar.",

    # Miscellaneous legitimate emails
    "Hi, I'm writing to confirm our dinner reservation for Friday at 7 PM. "
    "The restaurant is at 123 Main Street. "
    "They said they can accommodate dietary restrictions, just let them know in advance.",

    "Hey, the weather forecast looks great for this weekend. "
    "Want to go hiking at the national park? "
    "I was thinking we could take the scenic trail that's about 5 miles long.",

    "Good morning, I just wanted to thank you for helping me move last weekend. "
    "I really couldn't have done it without your help. "
    "Let me buy you lunch to say thanks.",

    "Hi, the landlord sent an email about the building maintenance schedule. "
    "They'll be fixing the elevator on Tuesday and the water will be off for 2 hours on Wednesday. "
    "Just wanted to give you a heads up.",

    "Hey, I found that old photo album we were looking for. "
    "There are some really great pictures from our college days in there. "
    "I'll bring it over when I visit next time.",
]

# ─── Variation Functions ──────────────────────────────────────────────────────

SPAM_GREETINGS = [
    "Dear Friend,", "Dear Winner,", "Dear Customer,", "Attention!",
    "Dear Valued Member,", "Hello,", "Dear User,", "Greetings!",
    "To Whom It May Concern,", "Dear Lucky One,", ""
]

HAM_GREETINGS = [
    "Hi,", "Hey,", "Hello,", "Good morning,", "Hi there,",
    "Hey there,", "Dear team,", "Hello everyone,", ""
]

SPAM_CLOSINGS = [
    "Click here now!", "Act immediately!", "Don't wait!",
    "Limited time only!", "Hurry!", "Reply ASAP!",
    "This is not a joke!", "100% guaranteed!", ""
]

HAM_CLOSINGS = [
    "Best regards,", "Thanks,", "Cheers,", "Best,",
    "Talk soon,", "Regards,", "Take care,", "See you soon,", ""
]

SPAM_EXTRA_PHRASES = [
    "No credit card required.",
    "This is NOT spam.",
    "You have been specially selected.",
    "Forward this to 10 friends.",
    "Unsubscribe by clicking here.",
    "This offer won't last long.",
    "Reply with YES to confirm.",
    "Call our toll-free number now.",
    "Wire transfer only.",
    "Send your bank details.",
    "Guaranteed results.",
    "Risk-free trial.",
    "As seen on TV.",
    "Doctor recommended.",
    "Buy now pay later.",
]

HAM_EXTRA_PHRASES = [
    "Let me know your thoughts.",
    "Looking forward to hearing from you.",
    "Hope this helps.",
    "Have a great day.",
    "See you at the meeting.",
    "Thanks for your time.",
    "I appreciate your help.",
    "Don't hesitate to reach out.",
    "Feel free to call me.",
    "Let's discuss this further.",
]


def generate_spam_email():
    """Generate a single spam email with variations."""
    greeting = random.choice(SPAM_GREETINGS)
    body = random.choice(SPAM_BODIES)
    closing = random.choice(SPAM_CLOSINGS)
    extra = random.choice(SPAM_EXTRA_PHRASES) if random.random() > 0.4 else ""

    parts = [p for p in [greeting, body, extra, closing] if p]
    email_text = " ".join(parts)

    # Random modifications for variety
    if random.random() > 0.7:
        email_text = email_text.upper()
    if random.random() > 0.8:
        email_text = email_text.replace("!", "!!!")
    if random.random() > 0.85:
        email_text += " " + "".join(random.choices(["!", "?", "$", "*"], k=random.randint(2, 5)))

    return email_text


def generate_ham_email():
    """Generate a single ham (legitimate) email with variations."""
    greeting = random.choice(HAM_GREETINGS)
    body = random.choice(HAM_BODIES)
    closing = random.choice(HAM_CLOSINGS)
    extra = random.choice(HAM_EXTRA_PHRASES) if random.random() > 0.5 else ""

    parts = [p for p in [greeting, body, extra, closing] if p]
    email_text = " ".join(parts)

    return email_text


def generate_dataset(total_emails=1200, spam_ratio=0.45):
    """
    Generate a balanced dataset of spam and ham emails.

    Args:
        total_emails: Total number of emails to generate.
        spam_ratio: Proportion of spam emails (0.0 to 1.0).

    Returns:
        List of (email_text, label) tuples.
    """
    num_spam = int(total_emails * spam_ratio)
    num_ham = total_emails - num_spam

    dataset = []

    print(f"Generating {num_spam} spam emails...")
    for _ in range(num_spam):
        dataset.append((generate_spam_email(), "spam"))

    print(f"Generating {num_ham} ham emails...")
    for _ in range(num_ham):
        dataset.append((generate_ham_email(), "ham"))

    random.shuffle(dataset)
    return dataset


def save_to_csv(dataset, filepath):
    """Save the dataset to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["email_text", "label"])
        writer.writerows(dataset)

    print(f"Dataset saved to {filepath}")
    print(f"Total emails: {len(dataset)}")
    spam_count = sum(1 for _, label in dataset if label == "spam")
    ham_count = len(dataset) - spam_count
    print(f"Spam: {spam_count} | Ham: {ham_count}")


if __name__ == "__main__":
    random.seed(42)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(data_dir, "emails.csv")

    dataset = generate_dataset(total_emails=1200, spam_ratio=0.45)
    save_to_csv(dataset, csv_path)
    print("\nDataset generation complete!")
