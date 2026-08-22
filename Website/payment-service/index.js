require("dotenv").config();

const express = require("express");
const cors = require("cors");
const Safepay = require("@sfpy/node-core");

const app = express();

app.use(cors());
app.use(express.json());

const safepay = Safepay(process.env.SAFEPAY_SECRET_KEY, {
    authType: "secret",
    host: "https://sandbox.api.getsafepay.com"
});

// SDK Info
console.log("SDK METHODS:");
console.log(Object.keys(safepay));

console.log("AUTH OBJECT:");
console.log(safepay.auth);

console.log("GUEST OBJECT:");
console.dir(safepay.guests, { depth: 5 });

if (safepay.auth) {
    console.log("AUTH KEYS:");
    console.log(Object.keys(safepay.auth));

    console.log("PASSPORT OBJECT:");
    console.log(safepay.auth.passport);
}

console.log("CHECKOUT OBJECT:");
console.log(safepay.checkout);

if (safepay.checkout) {
    console.log("CHECKOUT KEYS:");
    console.log(Object.keys(safepay.checkout));
}

app.post("/create-payment", async (req, res) => {
    try {

        // Step 1 - Create Payment Session
        const session = await safepay.payments.session.setup({
            merchant_api_key: process.env.SAFEPAY_PUBLIC_KEY,
            intent: "CYBERSOURCE",
            mode: "payment",
            entry_mode: "raw",
            currency: "PKR",
            amount: 10000,
            metadata: {
                order_id: "ORDER1001"
            },
            include_fees: false
        });

        console.log("SESSION:");
        console.log(session);

        // Step 2 - Create Passport Token
        const passport = await safepay.client.passport.create();

        console.log("PASSPORT:");
        console.log(passport);

        // Step 3 - Generate Checkout URL
        const checkout = safepay.checkout.createCheckoutUrl({
            env: "sandbox",
            tracker: session.data.tracker.token,
            tbt: passport.data,
            source: "hosted",
            redirect_url: "http://localhost:5173/payment-success",
            cancel_url: "http://localhost:5173/payment"

        });

        console.log("CHECKOUT URL:");
        console.log(checkout);

        res.json({
            success: true,
            checkout_url: checkout
        });

    } catch (err) {

        console.error("ERROR:");
        console.error(err);

        return res.status(500).json({
            success: false,
            error: err.message,
            stack: err.stack
        });
    }
});

app.listen(3001, () => {
    console.log("Payment Service Running");
});