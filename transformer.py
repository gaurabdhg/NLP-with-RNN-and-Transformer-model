class TransformerModel(nn.Module):
    def __init__(self, source_vocabulary_size, target_vocabulary_size,
                 d_model=256, pad_id=0, encoder_layers=3, decoder_layers=2,
                 dim_feedforward=1024, num_heads=8):
        # all arguments are (int)
        super().__init__()
        self.pad_id = pad_id

        self.embedding_src = nn.Embedding(
            source_vocabulary_size, d_model, padding_idx=pad_id)
        self.embedding_tgt = nn.Embedding(
            target_vocabulary_size, d_model, padding_idx=pad_id)

        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model, num_heads, encoder_layers, decoder_layers, dim_feedforward)
        self.encoder = self.transformer.encoder
        self.decoder = self.transformer.decoder
        self.linear = nn.Linear(d_model, target_vocabulary_size)

    def create_src_padding_mask(self,src):
        # input src of shape ()
        src_padding_mask = src.transpose(0, 1) == 0
        return src_padding_mask

    def create_tgt_padding_mask(self,tgt):
        # input tgt of shape ()
        tgt_padding_mask = tgt.transpose(0, 1) == 0
        return tgt_padding_mask


    def forward(self, src, tgt):
        """Forward function.
        Parameters:
          src: tensor of shape (sequence_length, batch, data dim)
          tgt: tensor of shape (sequence_length, batch, data dim)
        Returns:
          tensor of shape (sequence_length, batch, data dim)
        """
        
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)
        out = self.transformer(
            out, tgt, src_key_padding_mask=src_key_padding_mask,
            tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        
        out = self.linear(out)
        return out

    def forward_separate(self,src,tgt):
        src_key_padding_mask = self.create_src_padding_mask(src).to(DEVICE)
        tgt_key_padding_mask = self.create_tgt_padding_mask(tgt).to(DEVICE)
        memory_key_padding_mask = src_key_padding_mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[0]).to(DEVICE)

        tgt = self.embedding_tgt(tgt)
        tgt = self.pos_encoder(tgt)
        out = self.embedding_src(src)
        out = self.pos_encoder(out)

        # Encode the source sequence
        encoder_output = self.encoder(src,src_key_padding_mask=src_key_padding_mask)

        # Decode the target sequence
        decoder_output = self.decoder(tgt, encoder_output,
                                      tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        out = self.linear(decoder_output)

        return out

        
    def greedy_search(self, src, tgt, max_len=100, start_token=3, eos_token=2):
        self.eval()
        with torch.no_grad():
            src_padding_mask = self.create_src_padding_mask(src)
            tgt_padding_mask = self.create_tgt_padding_mask(tgt)
            memory_key_padding_mask=src_padding_mask
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[0]).to(DEVICE)

            tgt=self.embedding_tgt(tgt)
            tgt=self.pos_encoder(tgt)
            out=self.embedding_src(src)
            out=self.pos_encoder(out)

            enc_output = self.encoder(src)

            dec_input = torch.LongTensor([[start_token]] * src.shape[1]).to(DEVICE)

            pred = []

            for i in range(max_len):
                dec_output = self.decoder(dec_input, enc_output, tgt_padding_mask)
                pred_tokens = dec_output.argmax(dim=-1)

                current = [pred[j] + [pred_tokens[j].item()] for j in range(src.shape[1])]
                
                stop_criteria = [pred_tokens[j] == eos_token or dec_input.shape[1] > tgt.shape[1] for j in range(src.shape[1])]
                
                if all(stop_criteria):
                    break
                else:
                    dec_input = torch.cat((dec_input, pred_tokens), dim=1)

                pred = current

            return pred
