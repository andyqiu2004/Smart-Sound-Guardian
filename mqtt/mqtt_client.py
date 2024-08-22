import paho.mqtt.client as mqtt
import time
import threading
import json
import logging

logging.basicConfig(level=logging.INFO)


class MQTTClient:
    def __init__(self, client_id, broker, port, username=None, password=None):
        self.client = mqtt.Client(client_id)
        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.subscriptions = []
        self.connect_flag = False

        if username and password:
            self.client.username_pw_set(username, password)

        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.on_subscribe = self.on_subscribe
        self.client.on_publish = self.on_publish

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            self.connect_flag = True
        except Exception as e:
            logging.error(f"Connection failed: {e}")
            self.reconnect()

    def reconnect(self):
        while not self.connect_flag:
            try:
                logging.info("Reconnecting to MQTT broker...")
                self.client.reconnect()
                self.connect_flag = True
            except Exception as e:
                logging.error(f"Reconnect failed: {e}")
                time.sleep(5)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def subscribe(self, topic, qos=1):
        self.subscriptions.append((topic, qos))
        self.client.subscribe(topic, qos)

    def publish(self, topic, payload, qos=1, retain=False):
        self.client.publish(topic, json.dumps(payload), qos=qos, retain=retain)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT broker")
            for topic, qos in self.subscriptions:
                client.subscribe(topic, qos=qos)
        else:
            logging.error(f"Connection failed with code {rc}")

    def on_disconnect(self, client, userdata, rc):
        self.connect_flag = False
        if rc != 0:
            logging.warning("Unexpected disconnection.")
            self.reconnect()

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            logging.info(f"Received message on {msg.topic}: {payload}")
            self.handle_message(msg.topic, payload)
        except Exception as e:
            logging.error(f"Failed to process message: {e}")

    def handle_message(self, topic, payload):
        logging.info(f"Handling message from {topic}")
        # Custom logic for handling messages based on topic

    def on_subscribe(self, client, userdata, mid, granted_qos):
        logging.info(f"Subscribed with QoS {granted_qos}")

    def on_publish(self, client, userdata, mid):
        logging.info(f"Message published with mid {mid}")

    def run_forever(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.disconnect()


def start_mqtt_client(client_id, broker, port, username=None, password=None, topics=[]):
    mqtt_client = MQTTClient(client_id, broker, port, username, password)

    def mqtt_thread():
        mqtt_client.connect()
        for topic, qos in topics:
            mqtt_client.subscribe(topic, qos=qos)
        mqtt_client.run_forever()

    t = threading.Thread(target=mqtt_thread)
    t.start()
    return mqtt_client
