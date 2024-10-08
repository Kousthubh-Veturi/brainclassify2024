const express = require('express');
const sequelize = require('./config/db');  
const cors = require('cors');             
const authRoutes = require('./routes/authenticator'); 
const imageRoutes = require('./routes/imagator'); 

const app = express();

app.use(express.json());
app.use(cors());
app.use('/auth', authRoutes);
app.use('/image', imageRoutes);

sequelize.authenticate()
    .then(() => console.log('MySQL database connected...'))
    .catch(err => console.error('Error connecting to MySQL:', err));

sequelize.sync({ force: false }) 
    .then(() => console.log('Database models synced'))
    .catch(err => console.error('Error syncing database models:', err));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});


