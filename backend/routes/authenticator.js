const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const nodemailer = require('nodemailer');
const User = require('../models/usermodel');
const { JWT_SECRET, EMAIL_USER, EMAIL_PASS, EMAIL_HOST, EMAIL_PORT } = require('./config/config');

const router = express.Router();

const transporter = nodemailer.createTransport({
    host: EMAIL_HOST,
    port: EMAIL_PORT,
    secure: false, 
    auth: {
        user: EMAIL_USER,
        pass: EMAIL_PASS,
    },
});

router.post('/signup', async (req, res) => {
    const { name, email, password } = req.body;
    try {

        const userExists = await User.findOne({ where: { email } });
        if (userExists) return res.status(400).json({ message: "User already exists" });


        const hashedPassword = await bcrypt.hash(password, 10);

     
        const newUser = await User.create({
            name,
            email,
            password: hashedPassword,
            isEmailConfirmed: false,  
        });


        const token = jwt.sign({ userId: newUser.id }, JWT_SECRET, { expiresIn: '1d' });


        const confirmationUrl = `http://yourdomain.com/auth/confirm/${token}`;


        const mailOptions = {
            from: EMAIL_USER,
            to: newUser.email,
            subject: 'Email Confirmation',
            html: `<p>Please click the link below to confirm your email:</p>
                   <a href="${confirmationUrl}">${confirmationUrl}</a>`,
        };

        transporter.sendMail(mailOptions, (err, info) => {
            if (err) {
                console.error('Error sending confirmation email:', err);
                return res.status(500).json({ message: 'Error sending confirmation email' });
            }
            res.status(200).json({ message: 'Signup successful! Please check your email for confirmation.' });
        });
    } catch (err) {
        console.error('Signup error:', err);
        res.status(500).json({ message: 'Server error' });
    }
});


router.get('/confirm/:token', async (req, res) => {
    const { token } = req.params;
    try {

        const decoded = jwt.verify(token, JWT_SECRET);
        const user = await User.findByPk(decoded.userId);

        if (!user) return res.status(400).json({ message: 'Invalid token' });
        if (user.isEmailConfirmed) return res.status(400).json({ message: 'Email already confirmed' });


        user.isEmailConfirmed = true;
        await user.save();
        res.status(200).json({ message: 'Email confirmed successfully' });
    } catch (err) {
        console.error('Confirmation error:', err);
        res.status(400).json({ message: 'Invalid or expired token' });
    }
});


router.post('/login', async (req, res) => {
    const { email, password } = req.body;
    try {

        const user = await User.findOne({ where: { email } });
        if (!user) return res.status(400).json({ message: 'Invalid email or password' });


        if (!user.isEmailConfirmed) {
            return res.status(400).json({ message: 'Please confirm your email to log in' });
        }


        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) return res.status(400).json({ message: 'Invalid email or password' });


        const token = jwt.sign({ userId: user.id }, JWT_SECRET, { expiresIn: '1d' });

        res.status(200).json({ token });
    } catch (err) {
        console.error('Login error:', err);
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;