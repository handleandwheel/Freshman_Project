// code for freshman project
//
// use arduino MEGA
//
// information of transport message:
//   serial: USB
//   type: char
//   send 'a' when the transmission is begin
//   send one line at a time end with 's'(9 chars at a time)
//   send 't' when all the information is sent
//
// first version edited by liuyuming, at 2020/10/27 20:26


//pins
#define X_DIRECT 5
#define X_PULSE 10
#define Y_DIRECT 6
#define Y_PULSE 11
#define RELAY 7

//length of a single draw of number
#define STEP 15

//robot mode
#define RECEIVING_SUDOKU 0
#define WRITING_SUDOKU 1

//transmission mode
#define TRANSMISSION_BEGIN 1
#define TRANSMISSION_NOT_BEGIN 0

//move choice
#define MOVE_HALF 0.5
#define MOVE_BLOCK 3.0
#define MOVE_DRAW 1.0

char sudoku[82] = {'s'};  //'s' means the content ends
int pos = 0;  // from 0 to 80
int mode = RECEIVING_SUDOKU;
int if_transmission = TRANSMISSION_NOT_BEGIN;

void move_left(float times)
{
  for(int n = 0; n < STEP * times; ++n)
  { 
    digitalWrite(X_DIRECT,LOW);
    digitalWrite(X_PULSE,HIGH);
    delayMicroseconds(1500);
    digitalWrite(X_PULSE,LOW);
    delayMicroseconds(1500);
  }
  delay(100);
}

void move_right(float times)
{
  for(int n = 0; n < STEP * times; ++n)
  { 
    digitalWrite(X_DIRECT,HIGH);
    digitalWrite(X_PULSE,HIGH);
    delayMicroseconds(1500);
    digitalWrite(X_PULSE,LOW);
    delayMicroseconds(1500);
  }
  delay(100);
}

void move_up(float times)
{
  for(int n = 0; n < STEP * times; ++n)
  { 
    digitalWrite(Y_DIRECT,HIGH);
    digitalWrite(Y_PULSE,HIGH);
    delayMicroseconds(1500);
    digitalWrite(Y_PULSE,LOW);
    delayMicroseconds(1500);
  }
  delay(100);
}

void move_down(float times)
{
  for(int n = 0; n < STEP * times; ++n)
  {
    digitalWrite(Y_DIRECT,LOW);
    digitalWrite(Y_PULSE,HIGH);
    delayMicroseconds(1500);
    digitalWrite(Y_PULSE,LOW);
    delayMicroseconds(1500);
  }
  delay(100);
}

// HIGH means the relay is on, the pen is down
void pen_down() {digitalWrite(RELAY, HIGH); delay(200);}

// LOW means the realy is off, the pen is up
void pen_up() {digitalWrite(RELAY, LOW); delay(200);}

void write1()
{
  move_up(MOVE_DRAW); pen_down();
  move_down(MOVE_DRAW); move_down(MOVE_DRAW);
  pen_up(); move_up(MOVE_DRAW);
}

void write2()
{
  move_up(MOVE_DRAW); move_left(MOVE_HALF); pen_down();
  move_right(MOVE_DRAW); move_down(MOVE_DRAW); move_left(MOVE_DRAW); move_down(MOVE_DRAW); move_right(MOVE_DRAW);
  pen_up(); move_up(MOVE_DRAW); move_left(MOVE_HALF);
}

void write3()
{
  move_up(MOVE_DRAW); move_left(MOVE_HALF); pen_down();
  move_right(MOVE_DRAW); move_down(MOVE_DRAW); move_left(MOVE_DRAW); pen_up(); move_right(MOVE_DRAW); pen_down(); move_down(MOVE_DRAW); move_left(MOVE_DRAW);
  pen_up(); move_up(MOVE_DRAW); move_right(MOVE_HALF);
}

void write4()
{
  move_up(MOVE_DRAW); move_left(MOVE_HALF); pen_down();
  move_down(MOVE_DRAW); move_right(MOVE_DRAW); pen_up(); move_up(MOVE_DRAW); pen_down(); move_down(MOVE_DRAW); move_down(MOVE_DRAW);
  pen_up(); move_up(MOVE_DRAW); move_left(MOVE_HALF);
}

void write5()
{
  move_up(MOVE_DRAW); move_right(MOVE_HALF); pen_down();
  move_left(MOVE_DRAW); move_down(MOVE_DRAW); move_right(MOVE_DRAW); move_down(MOVE_DRAW); move_left(MOVE_DRAW);
  pen_up(); move_up(MOVE_DRAW); move_left(MOVE_HALF);
}

void write6()
{
  move_up(MOVE_DRAW); move_right(MOVE_HALF); pen_down();
  move_left(MOVE_DRAW); move_down(MOVE_DRAW); move_down(MOVE_DRAW); move_right(MOVE_DRAW); move_up(MOVE_DRAW); move_left(MOVE_DRAW);
  pen_up(); move_right(MOVE_HALF);
}

void write7()
{
  move_up(MOVE_DRAW); move_left(MOVE_HALF); pen_down();
  move_right(MOVE_DRAW); move_down(MOVE_DRAW); move_down(MOVE_DRAW);
  pen_up(); move_up(MOVE_DRAW); move_left(MOVE_HALF);
}

void write8()
{
  move_right(MOVE_HALF); pen_down();
  move_left(MOVE_DRAW); move_up(MOVE_DRAW); move_right(MOVE_DRAW); move_down(MOVE_DRAW); move_down(MOVE_DRAW); move_left(MOVE_DRAW); move_up(MOVE_DRAW);
  pen_up(); move_left(MOVE_HALF);
}

void write9()
{
  move_right(MOVE_HALF); pen_down();
  move_left(MOVE_DRAW); move_up(MOVE_DRAW); move_right(MOVE_DRAW); move_down(MOVE_DRAW); move_down(MOVE_DRAW); move_left(MOVE_DRAW);
  move_up(MOVE_DRAW); move_right(MOVE_HALF); pen_up();
}

//move to the next block
void move_next()
{
  ++pos;
  if(pos%8 == 0)
  {
    for(int n = 0; n < 8; ++n)
    {
      move_left(MOVE_BLOCK);
    }
    move_down(MOVE_BLOCK);
  }
  else
  {
    move_right(MOVE_BLOCK);
  }
}

//to attach the receiving number to the whole sudoku
void attach(char attachment[10], char origin[81])
{
  int i, j;
  for(i = 0; i < 82; ++i)
  {
    if(origin[i] == 's') break;
  }
  for(j = 0; j < 9; ++j)
  {
    origin[i + j] = attachment[j];
  }
  attachment[i + j] = 's';
}

//control the write
void write(char number)
{
  switch(number)
  {
  case '0': move_next(); break;
  case '1': write1(); move_next(); break;
  case '2': write2(); move_next(); break;
  case '3': write3(); move_next(); break;
  case '4': write4(); move_next(); break;
  case '5': write5(); move_next(); break;
  case '6': write6(); move_next(); break;
  case '7': write7(); move_next(); break;
  case '8': write8(); move_next(); break;
  case '9': write9(); move_next(); break;
  case 's': exit(0); break;
  default: break;
  }
}


void setup()
{
    pinMode(X_DIRECT,OUTPUT);
    pinMode(X_PULSE,OUTPUT);
    pinMode(Y_DIRECT,OUTPUT);
    pinMode(Y_PULSE,OUTPUT);
    pinMode(RELAY, OUTPUT);
    Serial.begin(9600);
}

void loop()
{
  if(RECEIVING_SUDOKU == mode)
  {
    if(TRANSMISSION_NOT_BEGIN == if_transmission)
    {
      if(Serial.peek() == 'a') if_transmission = TRANSMISSION_BEGIN;
    }
    else
    {
      char received_line[10] = {' '};
      if(Serial.available() >= 10)
      {
        Serial.readBytesUntil('s', received_line, 100);
        attach(received_line, sudoku);
      }
      if(Serial.peek() == 't')  //if receive the terminator
      {
        mode == WRITING_SUDOKU;
      }
    }
  }
  else if(WRITING_SUDOKU == mode)
  {
    write(sudoku[pos]);
  }
}
